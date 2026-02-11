import streamlit as st
import pandas as pd
import plotly.graph_objects as go

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

IH_PRICE_PER_M2 = 15000  # capped IH price (assumption)
PROF_FEE_RATE = 0.12     # % of (construction + DCs)

st.set_page_config(page_title="CPT RLV Calculator", layout="wide")
st.title("🏗️ Cape Town Redevelopment: RLV & IH Sensitivity")

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
):
    """
    Consistent assumptions:
    - Proposed GBA = land_area * ff * (1 + density_bonus)
    - DCs payable only on net increase (brownfield credit): max(0, proposed - existing)
    - IH requirement applied to the *net increase* (consistent with DC logic).
      If you want IH on total proposed, change ih_gba = proposed * ih_pct.
    - Sales revenue based on final proposed split: market = proposed - ih_gba; IH = ih_gba
    - Profit = profit_pct_gdv * GDV
    - Prof fees = PROF_FEE_RATE * (construction + DCs)
    - RLV = GDV - (construction + DCs + prof fees + profit)
    """

    proposed_gba = land_area_m2 * ff * (1 + density_bonus_pct / 100)

    net_increase = max(0.0, proposed_gba - existing_gba_m2)

    ih_gba = net_increase * (ih_pct / 100.0)
    ih_gba = min(ih_gba, proposed_gba)  # safety

    market_gba = proposed_gba - ih_gba

    # DC logic: only on market share of net increase (IH exempt), with PT discount on roads/transport portion
    market_gba_increase = net_increase - ih_gba

    disc = pt_discount(pt_zone_value)

    roads_dc = market_gba_increase * ROADS_TRANSPORT_PORTION * disc
    other_dc = market_gba_increase * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
    total_dc = roads_dc + other_dc

    total_potential_dc = net_increase * DC_BASE_RATE
    dc_savings = total_potential_dc - total_dc  # savings due to IH exemption + PT discount

    gdv = (market_gba * market_price_per_m2) + (ih_gba * ih_price_per_m2)

    construction_costs = proposed_gba * const_cost_per_m2
    hard_plus_dc = construction_costs + total_dc

    prof_fees = hard_plus_dc * PROF_FEE_RATE
    profit = gdv * profit_pct_gdv

    rlv = gdv - (hard_plus_dc + prof_fees + profit)

    return {
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
        "profit": profit,
        "rlv": rlv,
        "brownfield_credit": existing_gba_m2 > proposed_gba,
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

# Sensitivity Matrix (now uses same engine)
st.subheader("Sensitivity Analysis: IH % vs Density Bonus (consistent engine)")
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
        )
        row.append(f"R {tmp['rlv'] / 1_000_000:.1f}M")
    matrix_data.append(row)

df_matrix = pd.DataFrame(
    matrix_data,
    index=[f"{x}% IH" for x in ih_levels],
    columns=[f"{x}% Bonus" for x in bonus_levels],
)
st.table(df_matrix)
