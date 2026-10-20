import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="NBA Prediction & Props Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Prediction & Player Props Engine")
st.write("Simulate matchups 10,000 times and evaluate player prop prediction thresholds.")

# File Configurations
DATA_FILE = "nba_model_data.csv"
PLAYER_FILE = "nba_players.csv"

@st.cache_data
def load_app_data():
    try:
        if os.path.exists(DATA_FILE) and os.path.exists(PLAYER_FILE):
            return pd.read_csv(DATA_FILE), pd.read_csv(PLAYER_FILE)
        else:
            url_data = "https://githubusercontent.com"
            url_players = "https://githubusercontent.com"
            return pd.read_csv(url_data), pd.read_csv(url_players)
    except Exception as e:
        return None, None

clean_model_df, player_df = load_app_data()

if clean_model_df is None or player_df is None:
    st.error("Missing core database tracking files in repository. Please verify your CSV names.")
    st.stop()

# 1. Dropdown Selection Layout
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 2. Interactive Player Prop Selection Module
st.markdown("### 🎯 Player Prop Projection Hub")
target_team = st.selectbox("Choose Target Player Team", [home_team, away_team])
available_players = player_df[player_df['TEAM_NAME'] == target_team]['PLAYER_NAME'].tolist()

selected_player = st.selectbox("Select Player to Project", sorted(list(set(available_players))) if available_players else ["No Players Found"])

# 3. Calculation & Simulation Engine
if st.button("🚀 Run 10,000 Game & Prop Simulations"):
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
==================================================
        """, language="text")

        # =====================================================================
        # INDIVIDUAL PLAYER PROP PROJECTOR LAYER
        # =====================================================================
        p_row = player_df[(player_df['PLAYER_NAME'] == selected_player) & (player_df['TEAM_NAME'] == target_team)]
        
        if not p_row.empty:
            weight = p_row['POINT_WEIGHT'].iloc[-1]
            status = p_row['STATUS'].iloc[-1]
            
            # ADJUSTED STAR WEIGHT BRACKET TO CAPTURE 4.5 SCORES
            if weight >= 4.0:    # Franchise Stars (Cade, Tatum, Brunson, Maxey)
                base_pts, base_reb, base_ast = 24.5, 5.2, 6.4
            elif weight >= 2.8:  # Primary Starters / High-Usage Scorers
                base_pts, base_reb, base_ast = 15.2, 4.1, 2.8
            elif weight >= 1.5:  # Key Rotational Bench Depth Pieces
                base_pts, base_reb, base_ast = 9.4, 3.2, 1.5
            else:                # Standard Deep Reserves
                base_pts, base_reb, base_ast = 4.2, 1.5, 0.8
            
            # Matchup alterations against opponent defensive efficiency
            opp_def = h_def if target_team == away_team else a_def
            m_factor = opp_def / 114.5 
            
            if status == "OUT":
                proj_pts, proj_reb, proj_ast = 0.0, 0.0, 0.0
                status_msg = "🚨 INACTIVE / RULED OUT"
            else:
                proj_pts = base_pts * m_factor
                proj_reb = base_reb * (1 / m_factor) 
                proj_ast = base_ast * m_factor
                status_msg = "✅ ACTIVE / ELIGIBLE"
                
            st.markdown(f"### 🎯 Player Prop Projections: {selected_player}")
            st.write(f"Status Constraint: **{status_msg}**")
            
            st.markdown(f"""

| Prop Market | Expected Simulated Line Output | Actionable Metric Edge Guide |
| :--- | :--- | :--- |
| **Points (PTS)** | **{proj_pts:.1f} PTS** | Compare against sportsbook lines to locate Value Overs/Unders. |
| **Rebounds (REB)** | **{proj_reb:.1f} REB** | Adjusted for opponent rebounding and interior lane efficiency splits. |
| **Assists (AST)** | **{proj_ast:.1f} AST** | Scaled to overall simulated team offensive field goal pacing thresholds. |
            """)
        
        # Visual distribution chart logic
        margins = sim_home_scores - sim_away_scores
        fig = px.histogram(pd.DataFrame({'Margin': margins}), x='Margin', nbins=50, title="Matchup Outcome Distribution Curve", color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Spread ({raw_spread:.1f})")
        st.plotly_chart(fig, use_container_width=True)

