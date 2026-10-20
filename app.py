import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="NBA Game Prediction Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Game Prediction Engine")
st.write("Simulate any matchup 10,000 times using live rolling efficiency metrics.")

# File Configurations
DATA_FILE = "nba_model_data.csv"

@st.cache_data
def load_app_data():
    try:
        if os.path.exists(DATA_FILE):
            return pd.read_csv(DATA_FILE)
        else:
            url_data = "https://githubusercontent.com"
            return pd.read_csv(url_data)
    except Exception as e:
        return None

clean_model_df = load_app_data()

if clean_model_df is None:
    st.error("Missing core database tracking files in repository. Please verify your CSV names.")
    st.stop()

# 1. Dropdown Selection Layout
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 2. Calculation & Simulation Engine
if st.button("🚀 Run 10,000 Game Simulations"):
    home_data = clean_model_df[clean_model_df['TEAM_NAME'] == home_team]
    away_data = clean_model_df[clean_model_df['TEAM_NAME'] == away_team]
    
    if home_data.empty or away_data.empty:
        st.error("Error retrieving core team metrics.")
    else:
        # Isolate team efficiency bounds
        h_off = home_data['EWMA_OFF_RATING'].iloc[-1]
        h_def = home_data['EWMA_DEF_RATING'].iloc[-1]
        a_off = away_data['EWMA_OFF_RATING'].iloc[-1]
        a_def = away_data['EWMA_DEF_RATING'].iloc[-1]
        
        # 99.2 Baseline Pace standardizations
        projected_home_base = (((h_off + a_def) / 2) + 2.5) / 100 * 99.2
        projected_away_base = ((a_off + h_def) / 2) / 100 * 99.2

        np.random.seed(42)
        sim_home_scores = np.random.normal(projected_home_base, 11, 10000)
        sim_away_scores = np.random.normal(projected_away_base, 11, 10000)
        
        home_wins = np.sum(sim_home_scores > sim_away_scores)
        home_win_pct = (home_wins / 10000) * 100
        away_win_pct = 100 - home_win_pct
        
        avg_home_final, avg_away_final = np.mean(sim_home_scores), np.mean(sim_away_scores)
        projected_total = avg_home_final + avg_away_final
        raw_spread = avg_home_final - avg_away_final 
        spread_string = f"{home_team} -{raw_spread:.1f}" if raw_spread > 0 else f"{away_team} -{abs(raw_spread):.1f}"
            
        st.markdown("### 📊 Game Simulation Results")
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

        # Visual distribution chart logic
        margins = sim_home_scores - sim_away_scores
        fig = px.histogram(pd.DataFrame({'Margin': margins}), x='Margin', nbins=50, title="Matchup Outcome Distribution Curve", color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Spread ({raw_spread:.1f})")
        st.plotly_chart(fig, use_container_width=True)
