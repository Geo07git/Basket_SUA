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
        return round(m + s/60, 2)
    except:
        return np.nan

# 🧠 Predicție + RMSE pentru fiecare coloană
def predict_next_game(df):
    if len(df) < 3:
        return {c: "Insuficiente date" for c in df.columns}, None
    X = np.arange(len(df)).reshape(-1, 1)
    preds = {}
    rmses = {}
    for c in df.columns:
        model = RandomForestRegressor(random_state=0)
        model.fit(X, df[c])
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(df[c], y_pred))
        prediction = model.predict([[len(df)]])[0]
        preds[c] = f"{prediction:.1f}"
        rmses[c] = f"{rmse:.1f}"
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

    for col in ["PTS", "TRB", "AST", "PF"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df[["MIN", "PTS", "TRB", "AST", "PF"]].dropna()

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


# 🌐 Interfață Streamlit
st.set_page_config(page_title=" NBA/WNBA Predictions ", layout="centered")
st.title("🏀 NBA/WNBA Predictions")
league = st.selectbox("Select league:", ["NBA", "WNBA"]).lower()
players = nba_players if league == "nba" else wnba_players
player_name = st.selectbox("Select player:", list(players.keys()))
#debug = st.checkbox("🔎 Debug mode")

if player_name:
    pid = players[player_name]
    df_all = pd.concat([
        scrape_stats(generate_url(pid, league, y), league)#, debug)
        for y in [2024, 2025]
    ], ignore_index=True)

    if df_all.empty:
        st.error("⚠️ No valid data found.")
    else:
        st.subheader("📊 Recent statistics")
        st.dataframe(df_all.tail(5), use_container_width=True)

        preds, rmses = predict_next_game(df_all)

        st.subheader("🔮 Predict next match")
        st.table(pd.DataFrame([preds]))

        st.subheader("📉 RMSE (eroare model pe date istorice)")
        st.table(pd.DataFrame([rmses]))
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

        st.subheader("✅ Final prediction (interval)")
        st.table(pd.DataFrame([final_preds]))