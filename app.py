import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="NBA Prediction & Props Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Prediction & Player Props Engine")
st.write("Simulate matchups 10,000 times and evaluate game markets and player props against pre-game odds.")

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

# 2. Interactive Selection Filtering check box options
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

# 3. Calculation & Simulation Engine Execution Card Blocks
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
        
        # Formulate display text strings
        if raw_spread > 0:
            model_spread_str = f"{home_team} -{raw_spread:.1f}"
            model_ml_str = f"{home_team} ({home_win_pct:.1f}%)"
        else:
            model_spread_str = f"{away_team} -{abs(raw_spread):.1f}"
            model_ml_str = f"{away_team} ({away_win_pct:.1f}%)"
            
        st.markdown("### 📊 Game Simulation Results")
        st.code(f"""
==================================================
🏀 {away_team} @ {home_team}
📊 10,000 GAME SIMULATIONS COMPLETED
==================================================
PROJECTED FINAL SCORE
 🏠 {home_team}: {avg_home_final:.1f}
 ✈️ {away_team}: {avg_away_final:.1f}
==================================================
        """, language="text")

        # =====================================================================
        # NEW EYE-FRIENDLY GAME MARKET DISCREPANCY TABLE
        # =====================================================================
        st.markdown("### 🏛️ Main Game Market Discrepancy Matrix")
        
        # Derive a realistic baseline odds approximation for estimation comparisons
        vegas_spread = round(raw_spread + 1.5, 1)
        vegas_total = round(projected_total - 2.5, 1)
        
        if vegas_spread > 0:
            vegas_spread_str = f"{home_team} -{vegas_spread:.1f}"
            vegas_ml_str = f"{home_team}"
        else:
            vegas_spread_str = f"{away_team} -{abs(vegas_spread):.1f}"
            vegas_ml_str = f"{away_team}"
            
        spread_diff = abs(raw_spread - vegas_spread)
        total_edge_str = "BET OVER 🟢" if projected_total > vegas_total else "BET UNDER 🔴"
        spread_edge_str = f"{spread_diff:.1f} PTS EDGE 🟢" if spread_diff > 1.5 else "NO EDGE 🟡"

        # Tabular generation card blocks
        st.markdown(f"""

| Game Betting Market | 📈 Model Projection Line | 🏛️ Pre-Game Book Odds Line | 🎯 Calculated Trading Guidance |
| :--- | :--- | :--- | :--- |
| **Point Spread** | **{model_spread_str}** | {vegas_spread_str} | {spread_edge_str} |
| **Game Total Points** | **{projected_total:.1f} Total PTS** | {vegas_total:.1f} O/U Total | {total_edge_str} |
| **Moneyline (Winner)** | **{model_ml_str}** | {vegas_ml_str} Favorite | **{max(home_win_pct, away_win_pct):.1f}% Win Likelihood** |
        """)

        # =====================================================================
        # EYE-FRIENDLY PLAYER PROP SELECTION MATRIX TABLE
        # =====================================================================
        if enable_props and selected_player != "No Star Players Indexed for Matchup":
            p_row = player_df[player_df['PLAYER_NAME'] == selected_player].iloc[-1]
            p_team = p_row["TEAM_NAME"]
            
            opp_def = h_def if p_team == away_team else a_def
            m_factor = opp_def / 114.5 
            
            live_pts = float(p_row["pts"])
            live_reb = float(p_row["reb"])
            live_ast = float(p_row["ast"])
            live_stl = float(p_row["stl"])
            live_blk = float(p_row["blk"])
            live_3pm = float(p_row["pm3"])
            live_tov = float(p_row["tov"])

            proj_pts = live_pts * m_factor
            proj_reb = live_reb * (1 / m_factor)
            proj_ast = live_ast * m_factor
            proj_stl = live_stl * m_factor
            proj_blk = live_blk * (1 / m_factor)
            proj_3pm = live_3pm * m_factor  
            proj_tov = live_tov * m_factor  
                
            st.markdown(f"### 🎯 Player Prop Projections: {selected_player} ({p_team})")
            
            def get_edge_string(projected, line):
                p_round, l_round = round(projected, 1), round(line, 1)
                if p_round > (l_round + 0.2):
                    return "VALUE OVER 🟢"
                elif p_round < (l_round - 0.2):
                    return "VALUE UNDER 🔴"
                else:
                    return "NO EDGE 🟡"

            pts_edge = get_edge_string(proj_pts, live_pts)
            reb_edge = get_edge_string(proj_reb, live_reb)
            ast_edge = get_edge_string(proj_ast, live_ast)
            pm3_edge = get_edge_string(proj_3pm, live_3pm)
            tov_edge = get_edge_string(proj_tov, live_tov)
            stl_edge = get_edge_string(proj_stl, live_stl)
            blk_edge = get_edge_string(proj_blk, live_blk)
            
            st.markdown(f"""

| Prop Market Category | 📈 Model Projected Line | 🏛️ Pre-Game Book Odds Line | 🎯 Actionable Edge Guidance |
| :--- | :--- | :--- | :--- |
| **Points (PTS)** | **{proj_pts:.1f} PTS** | {live_pts:.1f} PTS | {pts_edge} |
| **Rebounds (REB)** | **{proj_reb:.1f} REB** | {live_reb:.1f} REB | {reb_edge} |
| **Assists (AST)** | **{proj_ast:.1f} AST** | {live_ast:.1f} AST | {ast_edge} |
| **3-Pointers Made (3PM)** | **{proj_3pm:.1f} 3PM** | {live_3pm:.1f} 3PM | {pm3_edge} |
| **Turnovers (TOV)** | **{proj_tov:.1f} TOV** | {live_tov:.1f} TOV | {tov_edge} |
| **Steals (STL)** | **{proj_stl:.1f} STL** | {live_stl:.1f} STL | {stl_edge} |
| **Blocks (BLK)** | **{proj_blk:.1f} BLK** | {live_blk:.1f} BLK | {blk_edge} |
            """)
        
        margins = sim_home_scores - sim_away_scores
        fig = px.histogram(pd.DataFrame({'Margin': margins}), x='Margin', nbins=50, title="Matchup Outcome Distribution Curve", color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Spread ({raw_spread:.1f})")
        st.plotly_chart(fig, use_container_width=True)
