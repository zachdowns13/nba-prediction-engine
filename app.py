import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="NBA Prediction Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Game Prediction Engine")
st.write("Simulate any matchup 10,000 times using live rolling metrics.")

DATA_URL = "https://githubusercontent.com"

@st.cache_data
def load_default_data():
    try:
        return pd.read_csv(DATA_URL)
    except Exception:
        try:
            return pd.read_csv("nba_model_data.csv")
        except Exception:
            return None

clean_model_df = load_default_data()

if clean_model_df is None:
    st.error("Could not find data file.")
    st.stop()

# 1. Dropdowns populated automatically from your CSV data
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 2. Automated Schedule Rest Check via the Dataset
home_data = clean_model_df[clean_model_df['TEAM_NAME'] == home_team]
away_data = clean_model_df[clean_model_df['TEAM_NAME'] == away_team]

if st.button("🚀 Run 10,000 Simulations"):
    if home_data.empty or away_data.empty:
        st.error("Error retrieving metrics.")
    else:
        # Detect if file uses advanced ratings or traditional raw points
        is_advanced = 'EWMA_OFF_RATING' in home_data.columns or 'OFF_RATING' in home_data.columns
        
        if is_advanced:
            h_off = home_data['EWMA_OFF_RATING'].iloc[-1] if 'EWMA_OFF_RATING' in home_data.columns else home_data['OFF_RATING'].iloc[-1]
            h_def = home_data['EWMA_DEF_RATING'].iloc[-1] if 'EWMA_DEF_RATING' in home_data.columns else home_data['DEF_RATING'].iloc[-1]
            a_off = away_data['EWMA_OFF_RATING'].iloc[-1] if 'EWMA_OFF_RATING' in away_data.columns else away_data['OFF_RATING'].iloc[-1]
            a_def = away_data['EWMA_DEF_RATING'].iloc[-1] if 'EWMA_DEF_RATING' in away_data.columns else away_data['DEF_RATING'].iloc[-1]
            
            # Pace conversion calculation
            projected_home_base = (((h_off + a_def) / 2) + 2.5) / 100 * 99.2
            projected_away_base = ((a_off + h_def) / 2) / 100 * 99.2
        else:
            # Safe Fallback: Handle traditional raw points correctly without shrinking the totals
            h_off = home_data['ROLLING_PTS_5'].iloc[-1] if 'ROLLING_PTS_5' in home_data.columns else home_data.iloc[-1, 1]
            h_def = home_data['ROLLING_OPP_PTS_5'].iloc[-1] if 'ROLLING_OPP_PTS_5' in home_data.columns else home_data.iloc[-1, 2]
            a_off = away_data['ROLLING_PTS_5'].iloc[-1] if 'ROLLING_PTS_5' in away_data.columns else away_data.iloc[-1, 1]
            a_def = away_data['ROLLING_OPP_PTS_5'].iloc[-1] if 'ROLLING_OPP_PTS_5' in away_data.columns else away_data.iloc[-1, 2]
            
            projected_home_base = ((h_off + a_def) / 2) + 2.5
            projected_away_base = (a_off + h_def) / 2
        
        # Run the 10,000 Simulations
        np.random.seed(42)
        sim_home_scores = np.random.normal(projected_home_base, 11, 10000)
        sim_away_scores = np.random.normal(projected_away_base, 11, 10000)
        
        # Compile Metrics
        home_wins = np.sum(sim_home_scores > sim_away_scores)
        home_win_pct = (home_wins / 10000) * 100
        away_win_pct = 100 - home_win_pct
        
        avg_home_final, avg_away_final = np.mean(sim_home_scores), np.mean(sim_away_scores)
        projected_total = avg_home_final + avg_away_final
        raw_spread = avg_home_final - avg_away_final 
        
        if raw_spread > 0:
            spread_string = f"{home_team} -{raw_spread:.1f}"
        else:
            spread_string = f"{away_team} -{abs(raw_spread):.1f}"
            
        # Display Text Results
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

        # 3. Create a visual chart of the margin of victory
        st.markdown("### 📈 Visual Margin Distribution Chart")
        margins = sim_home_scores - sim_away_scores
        chart_data = pd.DataFrame({'Margin of Victory': margins})
        
        fig = px.histogram(
            chart_data, 
            x='Margin of Victory', 
            nbins=50, 
            title=f"Distribution of Outcomes (Positive = {home_team} Wins, Negative = {away_team} Wins)",
            color_discrete_sequence=['#ff4b4b']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="white", annotation_text="Toss-up Line")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Projected Spread ({raw_spread:.1f})")
        
        st.plotly_chart(fig, use_container_width=True)


