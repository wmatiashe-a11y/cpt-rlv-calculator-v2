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

st.set_page_config(page_title="CPT RLV Calculator", layout="wide")
st.title("🏗️ Cape Town Redevelopment: RLV & IH Sensitivity")

# --- SIDEBAR INPUTS ---
st.sidebar.header("1. Site Parameters")
land_area = st.sidebar.number_input("Land Area (m2)", value=1000)
existing_gba = st.sidebar.number_input("Existing GBA on Site (m2)", value=200)
zoning_key = st.sidebar.selectbox("Zoning Preset", list(ZONING_PRESETS.keys()))
pt_zone = st.sidebar.selectbox("PT Zone (Parking/DC Discount)", ["Standard", "PT1", "PT2"])

st.sidebar.header("2. Policy Toggles")
ih_percent = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)

st.sidebar.header("3. Financials (ZAR)")
market_price = st.sidebar.number_input("Market Sales Price (per m2)", value=35000)
const_cost = st.sidebar.number_input("Construction Cost (per m2)", value=17500)
profit_margin = st.sidebar.slider("Target Profit Margin (%)", 10, 25, 20) / 100

# --- CALCULATION ENGINE ---
ff = ZONING_PRESETS[zoning_key]["ff"]
base_bulk = land_area * ff
total_proposed_gba = base_bulk * (1 + (density_bonus / 100))

# DC Logic
net_increase = max(0, total_proposed_gba - existing_gba)
ih_gba = net_increase * (ih_percent / 100)
market_gba_increase = net_increase - ih_gba

# PT Discount
pt_discount_factor = 1.0
if pt_zone == "PT1": pt_discount_factor = 0.8
if pt_zone == "PT2": pt_discount_factor = 0.5

roads_dc = market_gba_increase * ROADS_TRANSPORT_PORTION * pt_discount_factor
other_dc = market_gba_increase * (DC_BASE_RATE - ROADS_TRANSPORT_PORTION)
total_dc = roads_dc + other_dc

# Incentive Effect (Savings)
total_potential_dc = net_increase * DC_BASE_RATE
dc_savings = total_potential_dc - total_dc

# RLV Logic
total_market_gba = total_proposed_gba - (total_proposed_gba * (ih_percent / 100))
total_ih_gba = total_proposed_gba - total_market_gba

gdv = (total_market_gba * market_price) + (total_ih_gba * 15000) # Capped IH price
total_costs = (total_proposed_gba * const_cost) + total_dc
prof_fees = total_costs * 0.12
rlv = (gdv / (1 + profit_margin)) - total_costs - prof_fees

# --- UI DISPLAY ---
col1, col2 = st.columns([1, 1])

with col1:
    st.metric("Residual Land Value (RLV)", f"R {rlv:,.2f}")
    
    # Incentive Effect Card
    st.success(f"""
    **🌟 Incentive Effect**  
    The {ih_percent}% IH requirement and {pt_zone} selection has saved you:  
    ### R {dc_savings:,.2f}  
    *in Development Charges.*
    """)

    if existing_gba > total_proposed_gba:
        st.warning("⚠️ Existing bulk exceeds proposed bulk. No DCs payable (Brownfield Credit).")

with col2:
    # Waterfall Chart
    fig = go.Figure(go.Waterfall(
        name = "RLV Breakdown", orientation = "v",
        measure = ["relative", "relative", "relative", "relative", "total"],
        x = ["GDV", "Construction", "DCs", "Fees/Profit", "Residual Land"],
        y = [gdv, - (total_proposed_gba * const_cost), -total_dc, -(gdv * profit_margin), 0],
        connector = {"line":{"color":"rgb(63, 63, 63)"}},
    ))
    st.plotly_chart(fig, use_container_width=True)

# Sensitivity Matrix
st.subheader("Sensitivity Analysis: IH % vs Density Bonus")
matrix_data = []
for ih in [0, 10, 20, 30]:
    row = []
    for bonus in [0, 20, 40]:
        # Simplified RLV for matrix
        temp_gba = base_bulk * (1 + (bonus / 100))
        temp_mkt = temp_gba * (1 - (ih / 100))
        temp_ih = temp_gba - temp_mkt
        temp_gdv = (temp_mkt * market_price) + (temp_ih * 15000)
        temp_rlv = (temp_gdv / (1 + profit_margin)) - (temp_gba * const_cost)
        row.append(f"R {temp_rlv/1000000:.1f}M")
    matrix_data.append(row)

df_matrix = pd.DataFrame(matrix_data, 
                         index=["0% IH", "10% IH", "20% IH", "30% IH"], 
                         columns=["0% Bonus", "20% Bonus", "40% Bonus"])
st.table(df_matrix)
