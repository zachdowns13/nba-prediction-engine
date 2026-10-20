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

# OFFICIAL LEAGUE LOGO INDEX CDN MAPPING
LOGO_MAP = {
    "Atlanta Hawks": "https://nba.com",
    "Boston Celtics": "https://nba.com",
    "Cleveland Cavaliers": "https://nba.com",
    "New Orleans Pelicans": "https://nba.com",
    "Chicago Bulls": "https://nba.com",
    "Dallas Mavericks": "https://nba.com",
    "Denver Nuggets": "https://nba.com",
    "Golden State Warriors": "https://nba.com",
    "Houston Rockets": "https://nba.com",
    "Los Angeles Clippers": "https://nba.com",
    "Los Angeles Lakers": "https://nba.com",
    "Miami Heat": "https://nba.com",
    "Milwaukee Bucks": "https://nba.com",
    "Minnesota Timberwolves": "https://nba.com",
    "Brooklyn Nets": "https://nba.com",
    "New York Knicks": "https://nba.com",
    "Orlando Magic": "https://nba.com",
    "Indiana Pacers": "https://nba.com",
    "Philadelphia 76ers": "https://nba.com",
    "Phoenix Suns": "https://nba.com",
    "Portland Trail Blazers": "https://nba.com",
    "Sacramento Kings": "https://nba.com",
    "San Antonio Spurs": "https://nba.com",
    "Oklahoma City Thunder": "https://nba.com",
    "Toronto Raptors": "https://nba.com",
    "Utah Jazz": "https://nba.com",
    "Memphis Grizzlies": "https://nba.com",
    "Washington Wizards": "https://nba.com",
    "Detroit Pistons": "https://nba.com",
    "Charlotte Hornets": "https://nba.com"
}

# 1. Dropdown Selection Layout with Live Logos side-by-side
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
    if home_team in LOGO_MAP:
        st.image(LOGO_MAP[home_team], width=110)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)
    if away_team in LOGO_MAP:
        st.image(LOGO_MAP[away_team], width=110)

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

        # =====================================================================
        # CLEAN CARD LAYOUT SPEC BREAKDOWN MOCK (REPLACES UGLY TABLES)
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
                    return "🟢 VALUE OVER EDGE"
                elif p_round < (l_round - 0.2):
                    return "🔴 VALUE UNDER EDGE"
                else:
                    return "🟡 NO ADADVANTAGE LINE"

            # WE CONVERT THE MULTI-ATTRIBUTE MATRIX DIRECTLY INTO AN INFOGRAPHIC-STYLE GRID
            props_data = [
                {"Market": "Points (PTS)", "Proj": f"{proj_pts:.1f}", "Base": f"{live_pts:.1f}", "Edge": get_edge_string(proj_pts, live_pts)},
                {"Market": "Rebounds (REB)", "Proj": f"{proj_reb:.1f}", "Base": f"{live_reb:.1f}", "Edge": get_edge_string(proj_reb, live_reb)},
                {"Market": "Assists (AST)", "Proj": f"{proj_ast:.1f}", "Base": f"{live_ast:.1f}", "Edge": get_edge_string(proj_ast, live_ast)},
                {"Market": "3-Pointers (3PM)", "Proj": f"{proj_3pm:.1f}", "Base": f"{live_3pm:.1f}", "Edge": get_edge_string(proj_3pm, live_3pm)},
                {"Market": "Turnovers (TOV)", "Proj": f"{proj_tov:.1f}", "Base": f"{live_tov:.1f}", "Edge": get_edge_string(proj_tov, live_tov)},
                {"Market": "Steals (STL)", "Proj": f"{proj_stl:.1f}", "Base": f"{live_stl:.1f}", "Edge": get_edge_string(proj_stl, live_stl)},
                {"Market": "Blocks (BLK)", "Proj": f"{proj_blk:.1f}", "Base": f"{live_blk:.1f}", "Edge": get_edge_string(proj_blk, live_blk)}
            ]
            

