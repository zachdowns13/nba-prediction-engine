import streamlit as st
import pandas as pd
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="NBA Prediction Engine", page_icon="🏀", layout="centered")

st.title("🏀 NBA Game Prediction Engine")
st.write("Simulate any matchup 10,000 times using live 5-game rolling metrics.")

# 2. Data Sourcing 
DATA_URL = "https://githubusercontent.com"

@st.cache_data
def load_default_data():
    try:
        df = pd.read_csv(DATA_URL)
        return df
    except Exception:
        try:
            return pd.read_csv("nba_model_data.csv")
        except Exception:
            return None

# Sidebar layout for file management
st.sidebar.header("📁 Data Controls")
uploaded_file = st.sidebar.file_uploader("Upload your own nba_model_data.csv", type=["csv"])

if uploaded_file is not None:
    clean_model_df = pd.read_csv(uploaded_file)
    st.sidebar.success("Using your uploaded CSV file!")
else:
    clean_model_df = load_default_data()
    if clean_model_df is not None:
        st.sidebar.info("Automatically pulled live baseline stats from repository.")
    else:
        st.sidebar.error("Could not find nba_model_data.csv. Please upload your file manually.")
        st.stop()

# 3. Dropdown Selection Interface
team_list = sorted(clean_model_df['TEAM_NAME'].unique())

col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0 if len(team_list) > 0 else None)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1 if len(team_list) > 1 else None)

# 4. Simulation Engine Logic
if st.button("🚀 Run 10,000 Simulations"):
    # Extract historical tracking data rows
    home_data = clean_model_df[clean_model_df['TEAM_NAME'] == home_team]
    away_data = clean_model_df[clean_model_df['TEAM_NAME'] == away_team]
    
    if home_data.empty or away_data.empty:
        st.error("Error retrieving rolling statistics for selected teams.")
    else:
        # Isolate the most recent 5-game rolling averages
        home_pts = home_data['ROLLING_PTS_5'].iloc[-1]
        home_opp_pts = home_data['ROLLING_OPP_PTS_5'].iloc[-1]
        away_pts = away_data['ROLLING_PTS_5'].iloc[-1]
        away_opp_pts = away_data['ROLLING_OPP_PTS_5'].iloc[-1]
        
        # Mathematical baseline configuration
        league_avg_pts = 114.5  
        hga_bump = 2.5          
        
        # Expected base points calculation
        projected_home_base = ((home_pts + away_opp_pts) / 2) + hga_bump
        projected_away_base = (away_pts + home_opp_pts) / 2
        
        # Run Normal Distribution Simulations
        np.random.seed(42)
        sim_home_scores = np.random.normal(projected_home_base, 11, 10000)
        sim_away_scores = np.random.normal(projected_away_base, 11, 10000)
        
        # Compile Metrics
        home_wins = np.sum(sim_home_scores > sim_away_scores)
        home_win_pct = (home_wins / 10000) * 100
        away_win_pct = 100 - home_win_pct
        
        avg_home_final = np.mean(sim_home_scores)
        avg_away_final = np.mean(sim_away_scores)
        
        projected_total = avg_home_final + avg_away_final
        raw_spread = avg_home_final - avg_away_final 
        
        if raw_spread > 0:
            spread_string = f"{home_team} -{raw_spread:.1f}"
        else:
            spread_string = f"{away_team} -{abs(raw_spread):.1f}"
            
        # 5. Reader-Friendly Dashboard Output
        st.markdown("### 📊 Simulation Dashboard Results")
        st.code(f"""
==================================================
🏀 {away_team} @ {home_team}
📊 10,000 GAME SIMULATIONS COMPLETED
==================================================
PROJECTED FINAL SCORE
 🏠 {home_team}: {avg_home_final:.1f}
 ✈️ {away_team}: {avg_away_final:.1f}
--------------------------------------------------
MODEL MARKET PROJECTIONS
 🔹 Projected Spread: {spread_string}
 🔹 Projected Total Points: {projected_total:.1f}
--------------------------------------------------
SIMULATED WIN PROBABILITIES
 🏠 {home_team} Wins: {home_win_pct:.1f}%
 ✈️ {away_team} Wins: {away_win_pct:.1f}%
==================================================
        """, language="text")

