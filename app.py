import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from dataclasses import dataclass

# --- CONFIGURATION & ZONING PRESETS ---
ZONING_PRESETS = {
    "GR2 (Suburban)": {"ff": 1.0, "height": 15, "coverage": 0.6},
    "GR4 (Flats)": {"ff": 1.5, "height": 24, "coverage": 0.6},
    "MU1 (Mixed Use)": {"ff": 1.5, "height": 15, "coverage": 0.75},
    "MU2 (High Density)": {"ff": 4.0, "height": 25, "coverage": 1.0},
    "GB7 (CBD/High Rise)": {"ff": 12.0, "height": 60, "coverage": 1.0},
}

DC_BASE_RATE = 514.10  # Total DC per m2
ROADS_TRANSPORT_PORTION = 285.35

IH_PRICE_PER_M2 = 15000            # capped IH price (assumption)
DEFAULT_PROF_FEE_RATE = 0.12       # % of (construction + DCs) when fees are NOT included in build cost

st.set_page_config(page_title="CPT RLV Calculator", layout="wide")
st.title("🏗️ Cape Town Redevelopment: RLV & IH Sensitivity")

# -----------------------
# Heritage overlay model
# -----------------------
@dataclass(frozen=True)
class HeritageOverlay:
    enabled: bool
    bulk_reduction_pct: float      # interpreted as "BONUS suppression %" (override bonus)
    cost_uplift_pct: float         # increases construction costs
    fees_uplift_pct: float         # increases professional fees rate (ignored if build cost is all-in)
    profit_uplift_pct: float       # increases required profit % of GDV


def apply_heritage_overlay(
    density_bonus_pct: float,
    base_cost_sqm: float,
    base_fees_rate: float,
    base_profit_rate: float,
    overlay: HeritageOverlay,
) -> tuple[float, float, float, float]:
    """
    Returns:
      (adj_density_bonus_pct, adj_cost_sqm, adj_fees_rate, adj_profit_rate)

    Heritage behavior:
    - Suppresses density bonus (instead of scaling down bulk).
    - bulk_reduction_pct interpreted as "bonus suppression %".
    """
    if not overlay.enabled:
        return density_bonus_pct, base_cost_sqm, base_fees_rate, base_profit_rate

    adj_bonus = density_bonus_pct * (1.0 - overlay.bulk_reduction_pct / 100.0)
    adj_bonus = max(0.0, adj_bonus)

    adj_cost = base_cost_sqm * (1.0 + overlay.cost_uplift_pct / 100.0)
    adj_fees = base_fees_rate * (1.0 + overlay.fees_uplift_pct / 100.0)
    adj_profit = base_profit_rate * (1.0 + overlay.profit_uplift_pct / 100.0)
    return adj_bonus, adj_cost, adj_fees, adj_profit


# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Site Parameters")
land_area = st.sidebar.number_input("Land Area (m2)", value=1000.0, min_value=0.0, step=50.0)
existing_gba = st.sidebar.number_input("Existing GBA on Site (m2)", value=200.0, min_value=0.0, step=25.0)
zoning_key = st.sidebar.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()))
pt_zone = st.sidebar.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"])

st.sidebar.header("2. Policy Toggles")
ih_percent = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)

st.sidebar.header("3. Financials (ZAR)")
market_price = st.sidebar.number_input("Market Sales Price (per m2)", value=35000.0, min_value=0.0, step=500.0)
const_cost = st.sidebar.number_input("Construction Cost (per m2)", value=17500.0, min_value=0.0, step=250.0)
profit_margin = st.sidebar.slider("Target Profit (as % of GDV)", 10, 25, 20) / 100

# NEW: All-in toggle
st.sidebar.markdown("---")
build_cost_includes_fees = st.sidebar.checkbox(
    "Construction cost includes professional fees (all-in rate)",
    value=False,
    help=(
        "If enabled, the Construction Cost/m² is treated as an all-in rate that already includes "
        "professional fees. The model will set professional fees to R0 to avoid double counting."
    ),
)

# --- BUILT HERITAGE OVERLAY INPUTS ---
st.sidebar.header("4. Overlays")
st.sidebar.subheader("🏛️ Built Heritage Overlay")

heritage_enabled = st.sidebar.checkbox("Enable Built Heritage Overlay", value=False)

heritage_bonus_suppression = st.sidebar.slider(
    "Bonus suppression (%)",
    0, 100, 50,
    disabled=not heritage_enabled,
    help="0% = full bonus achievable; 100% = bonus fully blocked."
)
heritage_cost_uplift = st.sidebar.slider("Construction cost uplift (%)", 0, 40, 8, disabled=not heritage_enabled)

heritage_fees_uplift = st.sidebar.slider(
    "Professional fees uplift (%)",
    0, 40, 5,
    disabled=(not heritage_enabled) or build_cost_includes_fees,
    help="Disabled when build cost is all-in, because fees are not modelled as a separate line item."
)

heritage_profit_uplift = st.sidebar.slider("Profit requirement uplift (%)", 0, 40, 5, disabled=not heritage_enabled)

heritage_overlay = HeritageOverlay(
    enabled=heritage_enabled,
    bulk_reduction_pct=float(heritage_bonus_suppression),  # now "bonus suppression"
    cost_uplift_pct=float(heritage_cost_uplift),
    fees_uplift_pct=float(heritage_fees_uplift) if not build_cost_includes_fees else 0.0,
    profit_uplift_pct=float(heritage_profit_uplift),
)


# --- HELPERS ---
def pt_discount(pt_zone_value: str) -> float:
    if pt_zone_value == "PT1":
        return 0.8
    if pt_zone_value == "PT2":
        return 0.5
    return 1.0


def compute_model(
    land_area_m2: float,
    existing_gba_m2: float,
    ff: float,
    density_bonus_pct: float,
    ih_pct: float,
    pt_zone_value: str,
    market_price_per_m2: float,
    ih_price_per_m2: float,
    const_cost_per_m2: float,
    profit_pct_gdv: float,
    base_prof_fee_rate: float,
    overlay: HeritageOverlay,
    build_cost_includes_fees: bool,
):
    """
    Key rules:
    - Base bulk = land_area * ff
    - Heritage suppresses density bonus (override), not bulk scaling
    - Proposed GBA = base_bulk * (1 + effective_bonus)
    - IH applied to net increase (consistent with DC basis)
    - DCs payable on market share of net increase; PT discount on roads portion
    - Profit = (effective profit rate) * GDV
    - If build_cost_includes_fees: professional fees = 0 (avoid double-counting)
      else: prof fees = (effective fee rate) * (construction + DCs)
    - RLV = GDV - (construction + DCs + prof fees + profit)
    """

    base_bulk = land_area_m2 * ff

    # If build cost is all-in, fees are NOT added separately.
    fees_rate_input = 0.0 if build_cost_includes_fees else base_prof_fee_rate

    # Apply overlay (bonus override + cost/fees/profit uplifts)
    adj_bonus_pct, adj_cost_sqm, adj_fees_rate, adj_profit_rate = apply_heritage_overlay(
        density_bonus_pct=density_bonus_pct,
        base_cost_sqm=const_cost_per_m2,
        base_fees_rate=fees_rate_input,
        base_profit_rate=profit_pct_gdv,
        overlay=overlay,
    )

    proposed_gba = base_bulk * (1.0 + adj_bonus_pct / 100.0)

    # Brownfield credit / net increase
    net_increase = max(0.0, proposed_gba - existing_gba_m2)

    # IH applied to net increase (consistent with DC basis)
    ih_gba = net_increase * (ih_pct / 100.0)
    ih_gba = min(ih_gba, proposed_gba)

    market_gba = proposed_gba - ih_gba
    market_gba_increase = net_increase - ih_gba

    # DCs (market share only), PT discount on roads portion
    disc = pt_discount(pt_zone_value)
    roads_dc = market_gba_increase * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_gba_increase * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc

    # Revenue
    gdv = (market_gba * market_price_per_m2) + (ih_gba * ih_price_per_m2)

    # Costs
    construction_costs = proposed_gba * adj_cost_sqm
    hard_plus_dc = construction_costs + total_dc

    if build_cost_includes_fees:
        prof_fees = 0.0
        effective_fee_rate = 0.0
    else:
        prof_fees = hard_plus_dc * adj_fees_rate
        effective_fee_rate = adj_fees_rate

    profit = gdv * adj_profit_rate
    rlv = gdv - (hard_plus_dc + prof_fees + profit)

    return {
        "base_bulk": base_bulk,
        "input_bonus_pct": float(density_bonus_pct),
        "adj_bonus_pct": float(adj_bonus_pct),
        "proposed_gba": proposed_gba,
        "net_increase": net_increase,
        "ih_gba": ih_gba,
        "market_gba": market_gba,
        "market_gba_increase": market_gba_increase,
        "total_dc": total_dc,
        "dc_savings": dc_savings,
        "gdv": gdv,
        "construction_costs": construction_costs,
        "prof_fees": prof_fees,
        "effective_fee_rate": float(effective_fee_rate),
        "profit": profit,
        "rlv": rlv,
        "brownfield_credit": existing_gba_m2 > proposed_gba,
        "adj_cost_sqm": adj_cost_sqm,
        "adj_profit_rate": adj_profit_rate,
    }


# --- CALCULATION ENGINE ---
ff = ZONING_PRESETS[zoning_key]["ff"]

res = compute_model(
    land_area_m2=land_area,
    existing_gba_m2=existing_gba,
    ff=ff,
    density_bonus_pct=density_bonus,
    ih_pct=ih_percent,
    pt_zone_value=pt_zone,
    market_price_per_m2=market_price,
    ih_price_per_m2=IH_PRICE_PER_M2,
    const_cost_per_m2=const_cost,
    profit_pct_gdv=profit_margin,
    base_prof_fee_rate=DEFAULT_PROF_FEE_RATE,
    overlay=heritage_overlay,
    build_cost_includes_fees=build_cost_includes_fees,
)

# --- UI DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Residual Land Value (RLV)", f"R {res['rlv']:,.2f}")

    st.success(f"""
**🌟 Incentive Effect**  
With **{ih_percent}% IH** and **{pt_zone}**, you save:  
### R {res['dc_savings']:,.2f}  
*in Development Charges (vs charging full DCs on all net increase).*
""")

    if res["brownfield_credit"]:
        st.warning("⚠️ Existing GBA exceeds proposed GBA. Net increase is zero; no DCs payable (Brownfield Credit).")

    if build_cost_includes_fees:
        st.info("🧾 Professional fees are assumed to be included in the Construction Cost/m² (all-in). Fees line item set to R0 to avoid double-counting.")

    if heritage_overlay.enabled:
        st.info(f"""
**🏛️ Built Heritage Overlay Active**
- Bonus suppression: **{heritage_overlay.bulk_reduction_pct:.0f}%** → effective bonus **{res['adj_bonus_pct']:.1f}%** (input was {res['input_bonus_pct']:.1f}%)
- Cost uplift: **{heritage_overlay.cost_uplift_pct:.0f}%** → **R {res['adj_cost_sqm']:,.0f}/m²**
- Fees rate used: **{res['effective_fee_rate']*100:.2f}%** {'(all-in mode → 0%)' if build_cost_includes_fees else ''}
- Profit uplift: **{heritage_overlay.profit_uplift_pct:.0f}%** → **{res['adj_profit_rate']*100:.2f}%**
""")

    st.caption(
        f"Proposed GBA: {res['proposed_gba']:,.0f} m²  •  Net increase: {res['net_increase']:,.0f} m²  •  "
        f"IH: {res['ih_gba']:,.0f} m²  •  Market: {res['market_gba']:,.0f} m²"
    )

with col2:
    # Waterfall Chart (reconciles to RLV)
    fig = go.Figure(go.Waterfall(
        name="RLV Breakdown",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "relative", "total"],
        x=["GDV", "Construction", "DCs", "Professional Fees", "Profit", "Residual Land"],
        y=[
            res["gdv"],
            -res["construction_costs"],
            -res["total_dc"],
            -res["prof_fees"],
            -res["profit"],
            res["rlv"],
        ],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig.update_layout(margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

# Sensitivity Matrix (uses same engine + overlay + all-in toggle)
st.subheader("Sensitivity Analysis: IH % vs Density Bonus (overlay-consistent)")

ih_levels = [0, 10, 20, 30]
bonus_levels = [0, 20, 40]

matrix_data = []
for ih in ih_levels:
    row = []
    for bonus in bonus_levels:
        tmp = compute_model(
            land_area_m2=land_area,
            existing_gba_m2=existing_gba,
            ff=ff,
            density_bonus_pct=bonus,
            ih_pct=ih,
            pt_zone_value=pt_zone,
            market_price_per_m2=market_price,
            ih_price_per_m2=IH_PRICE_PER_M2,
            const_cost_per_m2=const_cost,
            profit_pct_gdv=profit_margin,
            base_prof_fee_rate=DEFAULT_PROF_FEE_RATE,
            overlay=heritage_overlay,
            build_cost_includes_fees=build_cost_includes_fees,
        )
        row.append(f"R {tmp['rlv'] / 1_000_000:.1f}M")
    matrix_data.append(row)

df_matrix = pd.DataFrame(
    matrix_data,
    index=[f"{x}% IH" for x in ih_levels],
    columns=[f"{x}% Bonus" for x in bonus_levels],
)
st.table(df_matrix)
