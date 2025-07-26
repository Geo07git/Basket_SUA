import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import os
import json

# 🎯 Convertire minute MM:SS → float
def convert_minutes(mp):
    try:
        m, s = map(int, mp.split(":"))
        return round(m + s / 60, 2)
    except:
        return np.nan

# 🧠 Predicție + RMSE pentru fiecare coloană (folosind RandomForest)
def predict_next_game(df):
    if len(df) < 3:
        return {c: "Insuficiente date" for c in df.columns}, None

    df = df.fillna(0)
    X = np.arange(len(df)).reshape(-1, 1)

    preds = {}
    rmses = {}

    for c in df.columns:
        try:
            y = df[c].astype(float).values
            if np.all(y == 0):  # Ignoră complet coloanele fără date semnificative
                preds[c] = 0.0
                rmses[c] = 0.0
                continue

            model = RandomForestRegressor(random_state=0)
            model.fit(X, y)
            y_pred = model.predict(X)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            prediction = model.predict([[len(df)]])[0]

            preds[c] = round(float(prediction), 1) if not np.isnan(prediction) else 0.0
            rmses[c] = round(float(rmse), 1) if not np.isnan(rmse) else 0.0
        except Exception as e:
            preds[c] = 0.0
            rmses[c] = 0.0

    return preds, rmses

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import streamlit as st

# 🎯 Funcție pentru a încărca datele NBA/WNBA pentru tot sezonul
@st.cache_data
def load_games(league, season):
    months = ["october", "november", "december", "january", "february", "march", "april"]
    all_games = []
    
    if league == "nba":
        # URL-ul de bază pentru scraping NBA
        base_url = f"https://www.basketball-reference.com/leagues/NBA_{season}_games-{{}}.html"
        
        for month in months:
            url = base_url.format(month)
            try:
                # Se trimite o cerere HTTP GET
                r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
                soup = BeautifulSoup(r.text, 'html.parser')
                # Găsește tabelul cu datele meciurilor
                table = soup.find("table", {"id": "schedule"})
                
                if table:
                    # Extrage datele în DataFrame
                    df = pd.read_html(StringIO(str(table)))[0]
                    df["Month"] = month  # Adăugăm luna pentru fiecare tabel
                    all_games.append(df)
            except Exception as e:
                st.error(f"⚠️ Nu s-au putut încărca datele pentru luna {month}. Detaliu: {e}")
        
        if all_games:
            # Combină toate lunile într-un singur DataFrame
            games_df = pd.concat(all_games, ignore_index=True)
            games_df = games_df.dropna(subset=["PTS", "PTS.1"])
            games_df = games_df.rename(columns={
                "Visitor/Neutral": "Away",
                "Home/Neutral": "Home",
                "PTS": "Away_PTS",
                "PTS.1": "Home_PTS"
            })
            games_df["Away_PTS"] = pd.to_numeric(games_df["Away_PTS"], errors='coerce')
            games_df["Home_PTS"] = pd.to_numeric(games_df["Home_PTS"], errors='coerce')
            games_df = games_df.dropna(subset=["Away_PTS", "Home_PTS"])
            return games_df
        else:
            return pd.DataFrame()
    
    elif league == "wnba":
        # URL-ul de bază pentru scraping WNBA
        url = f"https://www.basketball-reference.com/wnba/years/{season}_games.html"

        try:
            all_tables = pd.read_html(url)
            df = all_tables[0]
            df = df.dropna(subset=["PTS", "PTS.1"])
            df = df.rename(columns={
                "Visitor/Neutral": "Away",
                "Home/Neutral": "Home",
                "PTS": "Away_PTS",
                "PTS.1": "Home_PTS"
            })
            df["Away_PTS"] = pd.to_numeric(df["Away_PTS"], errors='coerce')
            df["Home_PTS"] = pd.to_numeric(df["Home_PTS"], errors='coerce')
            df = df.dropna(subset=["Away_PTS", "Home_PTS"])
            return df
        except Exception as e:
            st.error("⚠️ Nu s-au putut încărca datele")
            return pd.DataFrame()
    
    else:
        raise ValueError("League not recognized")

# Funcția pentru normalizarea datelor echipelor
def normalize_team_games(df):
    # Normalizarea echipelor ca "Home" și "Away"
    away_df = df[["Date", "Away", "Away_PTS", "Home", "Home_PTS"]].copy()
    away_df.columns = ["Date", "Team", "PTS", "Opponent", "Opponent_PTS"]
    away_df["Home"] = False  # Marcare pentru echipa de pe terenul oponentului

    home_df = df[["Date", "Home", "Home_PTS", "Away", "Away_PTS"]].copy()
    home_df.columns = ["Date", "Team", "PTS", "Opponent", "Opponent_PTS"]
    home_df["Home"] = True  # Marcare pentru echipa care joacă acasă

    # Combină ambele DataFrame-uri într-unul singur
    full_df = pd.concat([away_df, home_df], ignore_index=True)
    full_df["Point_Diff"] = full_df["PTS"] - full_df["Opponent_PTS"]
    return full_df

# 📊 Scrape date din BBR (inclusiv comentarii pentru NBA)
def scrape_stats(url, league, debug=False):
    headers = {'User-Agent': 'Mozilla/5.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, 'html.parser')

    table_id = "wnba_pgl_basic" if league == "wnba" else "player_game_log_reg"

    if league == "nba":
        comments = soup.find_all(string=lambda text: isinstance(text, Comment))
        for c in comments:
            if table_id in c:
                soup = BeautifulSoup(c, "html.parser")
                break

    tbl = soup.find("table", {"id": table_id})

    if debug:
        st.write(f"🔍 URL folosit: {url}")
        st.write(f"⚠️ Tabelul {'găsit ✅' if tbl else 'NU a fost găsit ❌'}")
        st.code(tbl.prettify()[:1000] if tbl else "Tabel inexistent", language="html")

    if not tbl:
        return pd.DataFrame()

    df = pd.read_html(StringIO(str(tbl)))[0]
    df = df[df["Rk"] != "Rk"].reset_index(drop=True)
    df["MIN"] = df["MP"].astype(str).apply(convert_minutes)

    for col in ["PTS", "TRB", "AST", "PF", "3P", "FG", "FT"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["MIN", "PTS", "TRB", "AST", "PF", "3P", "FG", "FT"]].dropna()

# 🔗 Generează URL-ul corect
def generate_url(pid, league, season):
    if league == "wnba":
        return f"https://www.basketball-reference.com/wnba/players/{pid[0]}/{pid}/gamelog/{season}/"
    else:
        return f"https://www.basketball-reference.com/players/{pid[0]}/{pid}/gamelog/{season}/"

# 📥 Încarcă fișierele JSON cu jucători NBA/WNBA
def load_players_from_file(filename):
    path = os.path.join(os.path.dirname(__file__), filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

nba_players = load_players_from_file("nba_active_players.json")
wnba_players = load_players_from_file("wnba_active_players.json")

# 📈 Analiza trend și consistență pentru fiecare coloană
def analyze_trend_consistency(df, method="ultimele", n=5):
    results = {}
    if method == "ultimele":
        df = df.tail(n)
        X = np.arange(len(df)).reshape(-1, 1)
        weights = None
    elif method == "ponderat":
        X = np.arange(len(df)).reshape(-1, 1)
        weights = np.exp(np.linspace(-1, 0, len(df)))

    std_all = {}

    for col in df.columns:
        y = df[col].values
        slope = 0
        try:
            if method == "ponderat":
                model = np.polyfit(X.flatten(), y, 1, w=weights)
            else:
                model = np.polyfit(X.flatten(), y, 1)
            slope = model[0]
        except:
            pass

        trend = "↗️ Crescător" if slope > 0.1 else "↘️ Descrescător" if slope < -0.1 else "➡️ Stabil"
        std_val = np.std(y)
        mean_val = np.mean(y)
        cv_val = (std_val / mean_val * 100) if mean_val != 0 else np.nan

        std_all[col] = std_val

        results[col] = {
            "Trend": trend,
            "STD (deviație)": std_val,
            "CV (% variabilitate)": cv_val
        }

    max_std = max(std_all.values()) if std_all else 1
    for col in results:
        std_val = std_all[col]
        score = 10 * (1 - (std_val / max_std)) if max_std != 0 else 10
        results[col]["Scor Consistență (0–10)"] = score

        results[col]["STD (deviație)"] = f"{results[col]['STD (deviație)']:.2f}"
        results[col]["CV (% variabilitate)"] = f"{results[col]['CV (% variabilitate)']:.2f}%"

    return pd.DataFrame(results).T

# Funcție pentru a calcula scorul final ajustat pe baza trendului și consistenței
def calculate_final_adjusted_score(pred, trend, std, cv):
    # Fără penalizări, doar un mic bonus/malus în funcție de trend
    trend_factor = 1.1 if trend == "↗️ Crescător" else 0.95 if trend == "↘️ Descrescător" else 1.0
    adjusted_pred = pred * trend_factor
    return round(adjusted_pred, 1)

# Funcție pentru a genera textul cu predicțiile finale ajustate
def generate_final_prediction_text(player_name, preds_adjusted):
    def fmt(val):
        return "0" if val in [None, np.nan, "nan", "-", "–"] or str(val).lower() == "nan" else str(val)

    player_name_upper = f"<span style='color:#FF5733; text-transform:uppercase; font-weight:bold'>{player_name}</span>"

    prediction_text = f"""
    <span style='color:#00FFAA; font-weight:bold; font-size:20px'>🏀 {player_name_upper}</span><br>
    ⏱️ va juca aproximativ <b>{fmt(preds_adjusted.get('MIN'))}</b> minute,<br>
    🎯 va înscrie <b>{fmt(preds_adjusted.get('PTS'))}</b> puncte,<br>
    🛡️ va avea <b>{fmt(preds_adjusted.get('TRB'))}</b> recuperări,<br>
    🎯 va pasa de <b>{fmt(preds_adjusted.get('AST'))}</b> ori.<br>
    💥 Aruncări de 3 puncte: <b>{fmt(preds_adjusted.get('3P'))}</b><br>
    🎯 Aruncări de la distanță: <b>{fmt(preds_adjusted.get('FG'))}</b><br>
    ⚠️ Faulturi estimate: <b>{fmt(preds_adjusted.get('PF'))}</b>
    """
    return prediction_text

# 🔢 Normalizează meciurile: fiecare rând = echipă + scor + adversar + acasă/deplasare
def normalize_team_games(df):
    away_df = df[["Date", "Away", "Away_PTS", "Home", "Home_PTS"]].copy()
    away_df.columns = ["Date", "Team", "PTS", "Opponent", "Opponent_PTS"]
    away_df["Home"] = False

    home_df = df[["Date", "Home", "Home_PTS", "Away", "Away_PTS"]].copy()
    home_df.columns = ["Date", "Team", "PTS", "Opponent", "Opponent_PTS"]
    home_df["Home"] = True

    full_df = pd.concat([away_df, home_df], ignore_index=True)
    full_df["Point_Diff"] = full_df["PTS"] - full_df["Opponent_PTS"]
    return full_df

# 🧠 Predicție diferență de puncte
def predict_margin(df, team1, team2):
    df_filtered = df[(df["Team"] == team1) | (df["Team"] == team2)].copy()
    df_filtered["Is_Team1"] = df_filtered["Team"] == team1

    X = pd.get_dummies(df_filtered[["Is_Team1", "Home"]], drop_first=True)
    y = df_filtered["Point_Diff"]

    model = LinearRegression()
    model.fit(X, y)

    # Predicție pentru meci pe teren neutru
    if location == "Echipa 1 acasă":
        input_data = pd.DataFrame([[1, 1]], columns=["Is_Team1", "Home"])
    elif location == "Teren neutru":
        input_data = pd.DataFrame([[1, 0]], columns=["Is_Team1", "Home"])
    else:  # Echipa 2 acasă
        input_data = pd.DataFrame([[0, 1]], columns=["Is_Team1", "Home"])
    pred = model.predict(input_data)
    return pred

# 🎯 Predicție total puncte în meci
def predict_total_points(df, team1, team2):
    df_matchups = df[((df["Team"] == team1) & (df["Opponent"] == team2)) |
                     ((df["Team"] == team2) & (df["Opponent"] == team1))]

    if len(df_matchups) >= 3:
        # Folosim media din meciurile directe dacă sunt suficiente
        avg_total = df_matchups["PTS"] + df_matchups["Opponent_PTS"]
        return avg_total.mean()
    else:
        # Altfel, folosim media punctelor din meciurile generale ale echipelor
        avg_team1 = df[df["Team"] == team1]["PTS"].mean()
        avg_team2 = df[df["Team"] == team2]["PTS"].mean()
        return avg_team1 + avg_team2


# 🚀 Interfață Streamlit
st.set_page_config(page_title="Predicții NBA/WNBA", layout="centered")
st.title("🏀 Predicții scor si jucatori NBA/WNBA")

# 📌 Selectori pe 2 coloane: Liga și Trend method
col1, col2 = st.columns(2)
with col1:
    league = st.selectbox("📂 Alege liga:", ["NBA", "WNBA"], key="league").lower()
with col2:
    trend_method = st.selectbox("📅 Alege metoda de analiză a trendului:", ["Trend ponderat recent", "Ultimele N meciuri"])

# Selecție echipe pentru predicția meciului
if league == "nba":
    games_df = load_games(league, 2025)  # Exemplu de sezon pentru NBA
else:
    games_df = load_games(league, 2025)  # Exemplu de sezon pentru WNBA

# Selecție echipe și locație pentru meciuri
if not games_df.empty:
    # Asigură-te că folosești DataFrame-ul corect după normalizare
    normalized_df = normalize_team_games(games_df)
    teams = sorted(normalized_df["Team"].unique())

    # 🏀 Selectare echipe - 2 coloane
    col3, col4 = st.columns(2)
    with col3:
        team1 = st.selectbox("🏠 Echipa gazdă", teams)
    with col4:
        team2 = st.selectbox("🚗 Echipa oaspete", teams)

    # 🏟️ Selectare locație - 3 coloane
    st.markdown("### 🏟️ Locația meciului")
    col5, col6, col7 = st.columns(3)
    with col5:
        loc1 = st.button("Echipa 1 acasă")
    with col6:
        loc2 = st.button("Echipa 2 acasă")
    with col7:
        loc3 = st.button("Teren neutru")

    # Determinăm locația pe baza butonului apăsat
    location = None
    if loc1:
        location = "Echipa 1 acasă"
    elif loc2:
        location = "Echipa 2 acasă"
    elif loc3:
        location = "Teren neutru"

    # 🔮 Predicții meci
    if team1 != team2:
        margin = predict_margin(normalized_df, team1, team2)
        total_points = predict_total_points(normalized_df, team1, team2)
        winner = team1 if margin > 0 else team2

        margin_val = margin[0]
        team1_points = (total_points + margin_val) / 2
        team2_points = (total_points - margin_val) / 2

        # 🔚 Rezumat final
        st.subheader("✅ Rezumat predicție meci")
        prediction_summary_html = f"""
        <div style='font-size:20px; background-color:#111; padding:20px; border-radius:10px; color:white; line-height:1.6'>
            🔮 <b><span style='color:#00FFAA'>{winner}</span></b> câștigă la o diferență de <b>{abs(margin_val):.0f}</b> puncte.<br>
            📊 <b>Scor estimat:</b> <span style='color:#FFD700'>{team1} {team1_points:.0f}</span> – <span style='color:#FFD700'>{team2_points:.0f} {team2}</span><br>
            🔢 <b>Total puncte:</b> {total_points:.1f}<br>
            🏟️ <b>Locație:</b> {location}
        </div>
        """
        st.markdown(prediction_summary_html, unsafe_allow_html=True)

# 🏀 Selecție jucător pentru predicții individuale
players = nba_players if league == "nba" else wnba_players
player_name = st.selectbox("Selectează jucătorul:", list(players.keys()))

# Preluare și predicție pentru jucător
if player_name:
    pid = players[player_name]
    df_all = pd.concat([scrape_stats(generate_url(pid, league, y), league) for y in [2024, 2025]], ignore_index=True)

    if df_all.empty:
        st.error("⚠️ Nu s-au găsit date valide.")
    else:
        # Calculează și afisează predicțiile pentru jucător
        if trend_method == "Ultimele N meciuri":
            n_last = st.slider("Număr de meciuri recente:", min_value=3, max_value=min(20, len(df_all)), value=5)
            trend_df = analyze_trend_consistency(df_all, method="ultimele", n=n_last)
        else:
            trend_df = analyze_trend_consistency(df_all, method="ponderat")

        preds, rmses = predict_next_game(df_all)

        # Calculăm predicțiile ajustate
        final_preds_adjusted = {}
        for key in preds:
            try:
                pred = float(preds[key])
                rmse = float(rmses[key])
                trend = trend_df.loc[key, 'Trend']
                std = float(trend_df.loc[key, 'STD (deviație)'])
                cv = float(trend_df.loc[key, 'CV (% variabilitate)'].replace('%', ''))

                adjusted_pred = calculate_final_adjusted_score(pred, trend, std, cv)
                final_preds_adjusted[key] = adjusted_pred
            except:
                final_preds_adjusted[key] = 0

        # Generăm textul final cu predicția
        prediction_text = generate_final_prediction_text(player_name, final_preds_adjusted)

        # Afișăm predicțiile
        st.subheader("✅ Rezumat predicție")
        st.markdown(
            f"<div style='font-size:17px; font-weight:bold; color:#00FFAA; background-color:black; padding:20px; border-radius:10px'>{prediction_text}</div>",
            unsafe_allow_html=True
        )

        st.subheader("🧱 Date folosite la predictii")

                # Optional: afișare date brute
        with st.expander("🔍 Vezi datele meciurilor analizate"):
            st.dataframe(normalized_df[(normalized_df["Team"] == team1) | (normalized_df["Team"] == team2)].sort_values("Date", ascending=False))

        with st.expander("📊 Recent statistics"):
            st.dataframe(df_all.tail(5), use_container_width=True)\
        
        def color_score(val):
            try:
                val = float(val)
                color = "limegreen" if val >= 7 else "gold" if val >= 4 else "tomato"
                return f"color: {color}; background-color: black; font-weight: bold;"
            except:
                return ""

        def color_cv(val):
            try:
                val = float(str(val).replace('%',''))
                color = "limegreen" if val < 20 else "gold" if val < 40 else "tomato"
                return f"color: {color}; background-color: black; font-weight: bold;"
            except:
                return ""

        def color_std(val):
            try:
                val = float(val)
                color = "limegreen" if val < 2 else "gold" if val < 4 else "tomato"    
                return f"color: {color}; background-color: black; font-weight: bold;"
            except:
                return ""

        with st.expander("🔍 Vezi datele trendului jucatorului"):
            st.dataframe(
            trend_df.style
                .map(color_score, subset=["Scor Consistență (0–10)"])
                .map(color_cv, subset=["CV (% variabilitate)"])
                .map(color_std, subset=["STD (deviație)"])
                .format(precision=2, subset=["Scor Consistență (0–10)"])
        )
