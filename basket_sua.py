import streamlit as st
import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup, Comment
from io import StringIO
from sklearn.ensemble import RandomForestRegressor
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

# 🌐 Interfață Streamlit
st.set_page_config(page_title="NBA/WNBA Player Predictions", layout="centered")
st.title("🏀 NBA/WNBA Player Predictions")

# 📌 Selectori pe 2 coloane: Liga și Trend method
col1, col2 = st.columns(2)
with col1:
    league = st.selectbox("📂 Alege liga:", ["NBA", "WNBA"], key="league").lower()
with col2:
    trend_method = st.selectbox("📅 Alege metoda de analiză a trendului:", ["Trend ponderat recent", "Ultimele N meciuri"])

#league = st.selectbox("Select league:", ["NBA", "WNBA"]).lower()
##league = st.radio("Alege liga:", ["NBA", "WNBA"], key="league").lower()
#trend_method = st.selectbox("Alege metoda de analiză a trendului:", ["Trend ponderat recent", "Ultimele N meciuri"])
##trend_method = st.radio("Alege metoda de analiză a trendului:", ["Ultimele N meciuri", "Trend ponderat recent"])
players = nba_players if league == "nba" else wnba_players
player_name = st.selectbox("Select player:", list(players.keys()))


if player_name:
    pid = players[player_name]
    df_all = pd.concat([scrape_stats(generate_url(pid, league, y), league) for y in [2024, 2025]], ignore_index=True)

    if df_all.empty:
        st.error("⚠️ No valid data found.")
    else:
        # 🔍 Calculam trend_df ÎNAINTE de predicțiile ajustate
        if trend_method == "Ultimele N meciuri":
            n_last = st.slider("Număr de meciuri recente:", min_value=3, max_value=min(20, len(df_all)), value=5)
            trend_df = analyze_trend_consistency(df_all, method="ultimele", n=n_last)
        else:
            trend_df = analyze_trend_consistency(df_all, method="ponderat")

        preds, rmses = predict_next_game(df_all)

        # 🧮 Final prediction interval
        final_preds = {}
        for key in preds:
            try:
                pred = float(preds[key])
                rmse = float(rmses[key])
                lower = round(pred - rmse, 1)
                upper = round(pred + rmse, 1)
                final_preds[key] = f"{lower} – {upper}"
            except:
                final_preds[key] = "–"

        # 🎯 Calculează predicțiile ajustate cu trend_df
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
                final_preds_adjusted[key] = 0  # în loc de "–", pentru textul final

        # ✨ Textul final cu predicția
        prediction_text = generate_final_prediction_text(player_name, final_preds_adjusted)

        # ✅ Afișăm și tabelul de predicție brută
        st.subheader("✅ Final prediction (interval)")
        st.table(pd.DataFrame([final_preds]))

        # 🔝 Afișează textul cu font mare, PRIMA IEȘIRE DUPĂ SELECTARE
        st.subheader("✅ Rezumat predictie")
        st.markdown(
            f"<div style='font-size:17px; font-weight:bold; color:#00FFAA; background-color:black; padding:20px; border-radius:10px'>{prediction_text}</div>",
            unsafe_allow_html=True
        )

        st.subheader("📊 Recent statistics")
        st.dataframe(df_all.tail(5), use_container_width=True)

        # 📉 Afișăm analiza trendului (colorată)
        st.subheader("📉 Analiză trend și consistență")

        def color_score(val):
            try:
                val = float(val)
                color = "limegreen" if val >= 7 else "gold" if val >= 4 else "tomato"
                return f"color: {color}; background-color: black; font-weight: bold;"
            except:
                return ""

        def color_cv(val):
            try:
                val = float(str(val).replace('%', ''))
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

        st.dataframe(
            trend_df.style
                .map(color_score, subset=["Scor Consistență (0–10)"])
                .map(color_cv, subset=["CV (% variabilitate)"])
                .map(color_std, subset=["STD (deviație)"])
                .format(precision=2, subset=["Scor Consistență (0–10)"])
        )
