import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

st.set_page_config(page_title="NBA Prediction & Props Engine", page_icon="🏀", layout="centered")
st.title("🏀 NBA Prediction & Player Props Engine")
st.write("Simulate matchups 10,000 times and evaluate player prop prediction thresholds against live lines.")

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

# ALL-INCLUSIVE LEAGUE-WIDE STAR DICTIONARY (With Added BLK/STL & Live Vegas Market Lines)
STAR_PROPS = {
    "Victor Wembanyama": {"pts": 25.0, "reb": 11.5, "ast": 3.0, "stl": 1.3, "blk": 3.6, "team": "San Antonio Spurs", "live_odds": "26.5 PTS / 11.5 REB / 2.5 AST / 1.5 STL / 3.5 BLK"},
    "Cade Cunningham": {"pts": 22.7, "reb": 4.3, "ast": 7.5, "stl": 1.0, "blk": 0.4, "team": "Detroit Pistons", "live_odds": "24.5 PTS / 5.5 REB / 9.5 AST / 1.0 STL / 0.5 BLK"},
    "Jayson Tatum": {"pts": 26.9, "reb": 8.1, "ast": 4.9, "stl": 1.0, "blk": 0.6, "team": "Boston Celtics", "live_odds": "27.5 PTS / 8.5 REB / 4.5 AST / 1.5 STL / 0.5 BLK"},
    "Jaylen Brown": {"pts": 23.0, "reb": 5.5, "ast": 3.6, "stl": 1.2, "blk": 0.5, "team": "Boston Celtics", "live_odds": "23.5 PTS / 5.5 REB / 3.5 AST / 1.0 STL / 0.5 BLK"},
    "Jalen Brunson": {"pts": 28.7, "reb": 3.6, "ast": 6.7, "stl": 0.9, "blk": 0.2, "team": "New York Knicks", "live_odds": "27.5 PTS / 3.5 REB / 6.5 AST / 1.0 STL / 0.5 BLK"},
    "Karl-Anthony Towns": {"pts": 21.8, "reb": 8.3, "ast": 3.0, "stl": 0.7, "blk": 0.7, "team": "New York Knicks", "live_odds": "20.5 PTS / 9.5 REB / 2.5 AST / 0.5 STL / 1.0 BLK"},
    "Tyrese Maxey": {"pts": 25.9, "reb": 3.7, "ast": 6.2, "stl": 1.0, "blk": 0.5, "team": "Philadelphia 76ers", "live_odds": "25.5 PTS / 3.5 REB / 5.5 AST / 1.0 STL / 0.5 BLK"},
    "LeBron James": {"pts": 25.7, "reb": 7.3, "ast": 8.3, "stl": 1.3, "blk": 0.5, "team": "Los Angeles Lakers", "live_odds": "23.5 PTS / 7.5 REB / 7.5 AST / 1.0 STL / 0.5 BLK"},
    "Anthony Davis": {"pts": 24.7, "reb": 12.6, "ast": 3.5, "stl": 1.2, "blk": 2.3, "team": "Los Angeles Lakers", "live_odds": "24.5 PTS / 12.5 REB / 3.5 AST / 1.5 STL / 2.5 BLK"},
    "Luka Doncic": {"pts": 33.9, "reb": 9.2, "ast": 9.8, "stl": 1.4, "blk": 0.5, "team": "Dallas Mavericks", "live_odds": "31.5 PTS / 9.5 REB / 8.5 AST / 1.5 STL / 0.5 BLK"},
    "Nikola Jokic": {"pts": 26.4, "reb": 12.4, "ast": 9.0, "stl": 1.4, "blk": 0.9, "team": "Denver Nuggets", "live_odds": "26.5 PTS / 12.5 REB / 8.5 AST / 1.5 STL / 1.0 BLK"},
    "Giannis Antetokounmpo": {"pts": 30.4, "reb": 11.5, "ast": 6.5, "stl": 1.2, "blk": 1.1, "team": "Milwaukee Bucks", "live_odds": "29.5 PTS / 11.5 REB / 5.5 AST / 1.0 STL / 1.0 BLK"},
    "Shai Gilgeous-Alexander": {"pts": 30.1, "reb": 5.5, "ast": 6.2, "stl": 2.0, "blk": 0.9, "team": "Oklahoma City Thunder", "live_odds": "29.5 PTS / 5.5 REB / 6.5 AST / 2.0 STL / 1.0 BLK"},
    "Stephen Curry": {"pts": 26.4, "reb": 4.5, "ast": 5.1, "stl": 0.7, "blk": 0.4, "team": "Golden State Warriors", "live_odds": "25.5 PTS / 4.5 REB / 5.5 AST / 1.0 STL / 0.5 BLK"},
    "Anthony Edwards": {"pts": 25.9, "reb": 5.4, "ast": 5.1, "stl": 1.3, "blk": 0.5, "team": "Minnesota Timberwolves", "live_odds": "26.5 PTS / 5.5 REB / 4.5 AST / 1.5 STL / 0.5 BLK"},
    "Ausar Thompson": {"pts": 8.8, "reb": 6.4, "ast": 1.9, "stl": 1.1, "blk": 0.9, "team": "Detroit Pistons", "live_odds": "9.5 PTS / 5.5 REB / 1.5 AST / 1.0 STL / 1.0 BLK"}
}

# 1. Dropdown Selection Layout
team_list = sorted(clean_model_df['TEAM_NAME'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Select HOME Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Select AWAY Team", team_list, index=1)

# 2. OPTIONAL Player Prop Selection Module Checkbox
st.markdown("### ⚙️ Option Controls")
enable_props = st.checkbox("Activate Individual Player Prop Analyzer", value=True)

selected_player = "No Star Players Indexed for Matchup"
if enable_props:
    st.markdown("### 🎯 Player Prop Projection Hub")
    valid_players = [name for name, data in STAR_PROPS.items() if data["team"] in [home_team, away_team]]
    selected_player = st.selectbox(
        "Select Player to Project", 
        sorted(valid_players) if valid_players else ["No Star Players Indexed for Matchup"]
    )

# 3. Calculation & Simulation Engine
if st.button("🚀 Run 10,000 Game Matchup Simulations"):
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
        # EXTENDED BLOCKS/STEALS PROP PROJECTOR LAYER WITH LIVE LINES
        # =====================================================================
        if enable_props and selected_player != "No Star Players Indexed for Matchup":
            base_data = STAR_PROPS[selected_player]
            p_team = base_data["team"]
            
            # Determine defensive matchup factor
            opp_def = h_def if p_team == away_team else a_def
            m_factor = opp_def / 114.5 
            
            # Perform Matchup-Efficiency Deflection Scaling
            proj_pts = base_data["pts"] * m_factor
            proj_reb = base_data["reb"] * (1 / m_factor)
            proj_ast = base_data["ast"] * m_factor
            proj_stl = base_data["stl"] * m_factor
            proj_blk = base_data["blk"] * (1 / m_factor) # Elite blocks scale inversely to offensive tempo
                
            st.markdown(f"### 🎯 Player Prop Edge Dashboard: {selected_player}")
            st.write(f"Live Vegas Line Baseline Reference: **{base_data['live_odds']}**")
            
            # Render multi-attribute spec breakdown as a structured Markdown table
            st.markdown(f"""

| Prop Market Category | 📈 Model Projected Line | 🏛️ Sportsbook Consensus Odds Line | 🎯 Trading Actionable Edge Guidance |
| :--- | :--- | :--- | :--- |
| **Points (PTS)** | **{proj_pts:.1f} PTS** | {base_data['live_odds'].split(' / ')[0]} | {"VALUE OVER 🟢" if proj_pts > float(base_data['live_odds'].split(' / ')[0].split(' ')[0]) else "VALUE UNDER 🔴"} |
| **Rebounds (REB)** | **{proj_reb:.1f} REB** | {base_data['live_odds'].split(' / ')[1]} | {"VALUE OVER 🟢" if proj_reb > float(base_data['live_odds'].split(' / ')[1].split(' ')[0]) else "VALUE UNDER 🔴"} |
| **Assists (AST)** | **{proj_ast:.1f} AST** | {base_data['live_odds'].split(' / ')[2]} | {"VALUE OVER 🟢" if proj_ast > float(base_data['live_odds'].split(' / ')[2].split(' ')[0]) else "VALUE UNDER 🔴"} |
| **Steals (STL)** | **{proj_stl:.1f} STL** | {base_data['live_odds'].split(' / ')[3]} | {"VALUE OVER 🟢" if proj_stl > float(base_data['live_odds'].split(' / ')[3].split(' ')[0]) else "VALUE UNDER 🔴"} |
| **Blocks (BLK)** | **{proj_blk:.1f} BLK** | {base_data['live_odds'].split(' / ')[4]} | {"VALUE OVER 🟢" if proj_blk > float(base_data['live_odds'].split(' / ')[4].split(' ')[0]) else "VALUE UNDER 🔴"} |
            """)
        
        # Visual distribution chart logic
        margins = sim_home_scores - sim_away_scores
        fig = px.histogram(pd.DataFrame({'Margin': margins}), x='Margin', nbins=50, title="Matchup Outcome Distribution Curve", color_discrete_sequence=['#ff4b4b'])
        fig.add_vline(x=0, line_dash="dash", line_color="white")
        fig.add_vline(x=raw_spread, line_color="yellow", annotation_text=f"Spread ({raw_spread:.1f})")
        st.plotly_chart(fig, use_container_width=True)

