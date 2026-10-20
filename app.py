import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="NBA Prediction & Props Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Prediction & Player Props Engine")
st.write("Simulate matchups 10,000 times and evaluate player prop prediction thresholds across 7 core categories.")

# File Configuration layouts
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
    except Exception:
        return None, None

clean_model_df, player_df = load_app_data()

if clean_model_df is None or player_df is None:
    st.error("Missing core database tracking files in repository. Please verify your CSV names.")
    st.stop()

# 1. Dropdown Matchup Selector Layout
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 2. Interactive Selection Filtering checkbox options
st.markdown("### ⚙️ Option Controls")
enable_props = st.checkbox("Activate Individual Player Prop Analyzer", value=True)

selected_player = "No Star Players Indexed for Matchup"
if enable_props:
    st.markdown("### 🎯 Player Prop Prediction Hub")
    valid_players = player_df[player_df['TEAM_NAME'].isin([home_team, away_team])]['PLAYER_NAME'].tolist()
    selected_player = st.selectbox(
        "Select Player to Project", 
        sorted(list(set(valid_players))) if valid_players else ["No Star Players Indexed for Matchup"]
    )

# 3. Calculation & Simulation Engine
if st.button("🚀 Run 10,000 Game Matchup Simulations"):
    home_data = clean_model_df[clean_model_df['TEAM_NAME'] == home_team]
    away_data = clean_model_df[clean_model_df['TEAM_NAME'] == away_team]
    
    if home_data.empty or away_data.empty:
        st.error("Error retrieving core team metrics.")
    else:
        h_off = home_data['EWMA_OFF_RATING'].iloc[-1]
        h_def = home_data['EWMA_DEF_RATING'].iloc[-1]
        a_off = away_data['EWMA_OFF_RATING'].iloc[-1]
        a_def = away_data['EWMA_DEF_RATING'].iloc[-1]
        
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
        # EXACT DECIMAL MATCHING VALUE ENGINE
        # =====================================================================
        if enable_props and selected_player != "No Star Players Indexed for Matchup":
            p_row = player_df[player_df['PLAYER_NAME'] == selected_player].iloc[-1]
            p_team = p_row["TEAM_NAME"]
            
            opp_def = h_def if p_team == away_team else a_def
            m_factor = opp_def / 114.5 
            
            # Pull exact float numbers straight from the spreadsheet columns
            live_pts = float(p_row["pts"]) if "pts" in p_row else 15.5
            live_reb = float(p_row["reb"]) if "reb" in p_row else 5.5
            live_ast = float(p_row["ast"]) if "ast" in p_row else 3.5
            live_stl = float(p_row["stl"]) if "stl" in p_row else 1.5
            live_blk = float(p_row["blk"]) if "blk" in p_row else 0.5
            live_3pm = float(p_row["pm3"]) if "pm3" in p_row else 1.5
            live_tov = float(p_row["tov"]) if "tov" in p_row else 2.5

            # Dynamic Projections Math
            proj_pts = live_pts * m_factor
            proj_reb = live_reb * (1 / m_factor)
            proj_ast = live_ast * m_factor
            proj_stl = live_stl * m_factor
            proj_blk = live_blk * (1 / m_factor)
            proj_3pm = live_3pm * m_factor  
            proj_tov = live_tov * m_factor  
                
            st.markdown(f"### 🎯 Player Prop Projections: {selected_player} ({p_team})")
            
            # High-precision trading edge logic tags
            def get_edge_string(projected, line):
                if projected > (line + 0.2):
                    return "VALUE OVER 🟢"
                elif projected < (line - 0.2):
                    return "VALUE UNDER 🔴"
                else:
                    return "NO EDGE 🟡"

            # Render structured Multi-Attribute Breakdown Matrix Table
            st.markdown(f"""

| Prop Market Category | 📈 Model Projected Line | 🏛️ Live Sportsbook Line Baseline | 🎯 Trading Actionable Edge Guidance |
| :--- | :--- | :--- | :--- |
| **Points (PTS)** | **{proj_pts:.1f} PTS** | {live_pts:.1f} PTS | {get_edge_string(proj_pts, live_pts)} |
| **Rebounds (REB)** | **{proj_reb:.1f} REB** | {live_reb:.1f} REB | {get_edge_string(proj_reb, live_reb)} |
| **Assists (AST)** | **{proj_ast:.1f} AST** | {live_ast:.1f} AST | {get_edge_string(proj_ast, live_ast)} |
| **3-Pointers Made (3PM)** | **{proj_3pm:.1f} 3PM** | {live_3pm:.1f} 3PM | {get_edge_string(proj_3pm, live_3pm)} |
| **Turnovers (TOV)** | **{proj_tov:.1f} TOV** | {live_tov:.1f} TOV | {get_edge_string(proj_tov, live_tov)} |
| **Steals (STL)** | **{proj_stl:.1f} STL** | {live_stl:.1f} STL | {get_edge_string(proj_stl, live_stl)} |
| **Blocks (BLK)** | **{proj_blk:.1f} BLK** | {live_blk:.1f} BLK | {get_edge_string(proj_blk, live_blk)} |
            """)
        
        margins = sim_home_scores - sim_away_scores
        fig = px.histogram(pd.DataFrame({'Margin': margins}), x='Margin', nbins=50, title="Matchup Outcome Distribution Curve", color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Spread ({raw_spread:.1f})")
        st.plotly_chart(fig, use_container_width=True)
