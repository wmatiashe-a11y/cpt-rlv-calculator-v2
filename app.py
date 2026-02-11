import streamlit as st
import pandas as pd
import plotly.express as px

# --- CAPE TOWN DATA PRESETS ---
ZONING = {
    "GR2 (Residential)": {"ff": 1.0, "coverage": 0.6, "height": 15},
    "GR4 (High Density Res)": {"ff": 1.5, "coverage": 0.6, "height": 24},
    "MU1 (Mixed Use)": {"ff": 1.5, "coverage": 0.75, "height": 15},
    "MU2 (High Density Mixed)": {"ff": 4.0, "coverage": 1.0, "height": 25},
}

DC_RATE = 514.10  # ZAR per m2 (Market units)
IH_CAP_PRICE = 15000  # ZAR per m2 for affordable units

st.set_page_config(page_title="CPT Land Value Calculator", layout="wide")
st.title("🏗️ Cape Town Residual Land Value Calculator")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Site Parameters")
land_size = st.sidebar.number_input("Land Area (m2)", value=1000)
zone_choice = st.sidebar.selectbox("Zoning Preset", list(ZONING.keys()))
market_price = st.sidebar.slider("Market Sales Price (R/m2)", 25000, 65000, 45000)
const_cost = st.sidebar.slider("Construction Cost (R/m2)", 14000, 25000, 17500)

st.sidebar.header("Policy Toggles")
ih_req = st.sidebar.slider("Inclusionary Housing (%)", 0, 30, 20)
density_bonus = st.sidebar.slider("Density Bonus (%)", 0, 50, 20)

# --- CALCULATION ENGINE ---
def calculate_rlv(land, ff, bonus, ih, m_price, c_cost):
    total_bulk = (land * ff) * (1 + (bonus / 100))
    ih_bulk = total_bulk * (ih / 100)
    market_bulk = total_bulk - ih_bulk
    
    # GDV
    gdv = (market_bulk * m_price) + (ih_bulk * IH_CAP_PRICE)
    
    # Costs
    dev_charges = market_bulk * DC_RATE # IH units are exempt
    construction = total_bulk * c_cost
    fees = construction * 0.12
    profit = gdv * 0.20
    
    rlv = gdv - construction - dev_charges - fees - profit
    return max(0, rlv)

# --- RESULTS ---
ff = ZONING[zone_choice]["ff"]
current_rlv = calculate_rlv(land_size, ff, density_bonus, ih_req, market_price, const_cost)

col1, col2, col3 = st.columns(3)
col1.metric("Residual Land Value", f"R {current_rlv:,.0f}")
col2.metric("Total Bulk (GBA)", f"{land_size * ff * (1 + density_bonus/100):,.0f} m2")
col3.metric("DC Savings (IH Incentive)", f"R {(land_size * ff * (ih_req/100)) * DC_RATE:,.0f}")

# --- SENSITIVITY MATRIX ---
st.subheader("Sensitivity Analysis: IH Requirement vs Density Bonus")
matrix_data = []
ih_range = [0, 10, 20, 30]
bonus_range = [0, 10, 20, 30, 40, 50]

for ih in ih_range:
    row = []
    for b in bonus_range:
        val = calculate_rlv(land_size, ff, b, ih, market_price, const_cost)
        row.append(round(val / 1000000, 2)) # In Millions
    matrix_data.append(row)

df = pd.DataFrame(matrix_data, index=[f"{i}% IH" for i in ih_range], 
                  columns=[f"+{b}% Bonus" for b in bonus_range])

st.table(df.style.background_gradient(cmap='RdYlGn'))
st.caption("Values in ZAR Millions. Green indicates higher land value.")
