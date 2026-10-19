import streamlit as st
import pandas as pd
import numpy as np
import plotly.figure_factory as ff

# 1. Page Configuration & Theme
st.set_page_config(page_title="Pro NBA Prediction Engine", page_icon="🏀", layout="centered")
st.title("🏀 Pro NBA Game Prediction Engine")
st.write("Simulating matchups 10,000 times using Efficiency, Fatigue, Injuries, and Visual Analytics.")

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

st.sidebar.header("📁 Data Controls")
uploaded_file = st.sidebar.file_uploader("Upload your advanced nba_model_data.csv", type=["csv"])

if uploaded_file is not None:
    clean_model_df = pd.read_csv(uploaded_file)
else:
    clean_model_df = load_default_data()
    if clean_model_df is None:
        st.error("Could not find data file.")
        st.stop()

# 2. Team Selection Dropdowns
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 3. ADVANCED TOOL #1: Schedule Fatigue Condition Check
st.markdown("### 🏃‍♂️ Schedule Fatigue Condition Check")
col3, col4 = st.columns(2)
with col3:
    home_b2b = st.checkbox(f"Is {home_team} on a Back-to-Back? (0 Days Rest)", value=False)
with col4:
    away_b2b = st.checkbox(f"Is {away_team} on a Back-to-Back? (0 Days Rest)", value=False)

# 4. ADVANCED TOOL #2: Live Roster Strength Injury Sliders
st.markdown("### 🚑 Live Roster Strength (Account for Injuries)")
st.write("Lower the slider if a star player (e.g., Tatum, Cunningham) is ruled out tonight.")
col5, col6 = st.columns(2)
with col5:
    home_health = st.slider(f"{home_team} Roster Strength %", min_value=50, max_value=100, value=100, step=1)
with col6:
    away_health = st.slider(f"{away_team} Roster Strength %", min_value=50, max_value=100, value=100, step=1)

# 5. Simulation Engine Logic
if st.button("🚀 Run 10,000 Pro Simulations"):
    home_data = clean_model_df[clean_model_df['TEAM_NAME'] == home_team]
    away_data = clean_model_df[clean_model_df['TEAM_NAME'] == away_team]
    
    if home_data.empty or away_data.empty:
        st.error("Error retrieving advanced metrics.")
    else:
        # Pull advanced exponential traits
        h_off, h_def = home_data['EWMA_OFF_RATING'].iloc[-1], home_data['EWMA_DEF_RATING'].iloc[-1]
        a_off, a_def = away_data['EWMA_OFF_RATING'].iloc[-1], away_data['EWMA_DEF_RATING'].iloc[-1]
        
        # Apply fatigue modifications
        h_b2b = 1.0 if home_b2b else 0.0
        a_b2b = 1.0 if away_b2b else 0.0
        
        # Factor in the custom roster strength percentages from sliders
        h_health_modifier = home_health / 100.0
        a_health_modifier = away_health / 100.0
        
        # Adjust base offensive ratings for fatigue and injury hits
        projected_home_rating = (((h_off * h_health_modifier) + a_def) / 2) + 2.5 - (h_b2b * 2.1)
        projected_away_rating = (((a_off * a_health_modifier) + h_def) / 2) - (a_b2b * 2.1)
        
        # Translate back into points per possession (using modern 99.2 standard pace baseline)
        projected_home_base = (projected_home_rating / 100) * 99.2
        projected_away_base = (projected_away_rating / 100) * 99.2
        
        # Run Normal Distribution Simulations
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
            
        # Display Text Dashboard
        st.markdown("### 📊 Advanced Simulation Results")
        st.code(f"""
==================================================
🏀 {away_team} @ {home_team}
📊 10,000 PRO GAME SIMULATIONS COMPLETED
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
        
        # ADVANCED TOOL #3: Interactive Plotly Probability Curve
        st.markdown("### 📈 Visual Score Probability Distribution")
        st.write("This chart displays the overlapping probability curves for both final scores across all 10,000 simulations.")
        
        hist_data = [sim_home_scores, sim_away_scores]
        group_labels = [f'{home_team} Projected Scores', f'{away_team} Projected Scores']
        colors = ['#ff4b4b', '#1f77b4'] # Red for home, Blue for away
        
        # Create the beautiful distribution graph using Plotly
        fig = ff.create_distplot(hist_data, group_labels, show_hist=False, show_rug=False, colors=colors)
        fig.update_layout(xaxis_title="Final Score Points", yaxis_title="Probability Density", legend_title="Teams")
        st.plotly_chart(fig, use_container_width=True)


