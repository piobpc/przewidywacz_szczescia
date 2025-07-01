# -*- coding: utf-8 -*-
"""
Aplikacja: „Przewidywacz poziomu szczęścia”
"""
import streamlit as st
import pandas as pd
import boto3
from botocore.config import Config
import os
from typing import Dict
from openai import OpenAI                                            
from pycaret.regression import load_model, predict_model  
import requests, textwrap
from typing import Optional

# ── 1. Usuń klucz środowiskowy, jeśli był ustawiony globalnie
os.environ.pop("OPENAI_API_KEY", None)

# ── 2.  Nie ładujemy już .env w produkcji, bo klucze są w systemie
#load_dotenv()
#env = dotenv_values(".env")         

# ── 3. Inicjalizacja stanu
if "openai_api_key" not in st.session_state:
    st.session_state.openai_api_key = ""

# ── 4. Formularz podawania klucza
def valid_key(k: str) -> bool:
    """Prosta walidacja: zaczyna się od sk- i ma ≥ 40 znaków."""
    return k.strip().startswith("sk-") and len(k.strip()) >= 40

if not valid_key(st.session_state.openai_api_key):
    with st.form("key_form", clear_on_submit=True):
        key_input = st.text_input(
            "🔑 Wpisz swój klucz OpenAI (format sk-…)",
            type="password",
            placeholder="sk-........................................",
        )
        submitted = st.form_submit_button("Zapisz klucz")
        if submitted:
            if valid_key(key_input):
                st.session_state.openai_api_key = key_input.strip()
                st.success("✅ Klucz przyjęty. Możesz korzystać z aplikacji.")
            else:
                st.error("❌ To nie wygląda na poprawny klucz OpenAI.")
    st.stop()   # ⟵ zatrzymujemy dalsze ładowanie aplikacji

# ── 5. Zainicjuj klienta – od tego miejsca klucz jest poprawny
openai_client = OpenAI(api_key=st.session_state.openai_api_key)

# pobieranie pozostałych kluczy
access_key  = os.getenv("AWS_ACCESS_KEY")
secret_key  = os.getenv("AWS_SECRET_KEY")
endpoint    = os.getenv("AWS_ENDPOINT_URL_S3")
BUCKET_NAME = os.getenv("AWS_BUCKET_NAME", "projectpiob")
OPENAI_MODEL = 'gpt-4o-mini'
MY_MODEL_TITLE_AWS = "whr_prediction_model_final" # można łatwo podmieniać modele
MY_MODEL_PATH_AWS = "models/whr_prediction_model_final.pkl" # j.w. 
  
missing = [n for n, v in {
    "AWS_ACCESS_KEY": access_key,
    "AWS_SECRET_KEY": secret_key,
    "AWS_ENDPOINT_URL_S3": endpoint,                                        
}.items() if not v]
if missing:
    st.error(f"❌ Brak zmiennych w .env: {', '.join(missing)}")
    st.stop()

# ---------- 2. KLIENT S3 i ŁADOWANIE MODELU ------------------
@st.cache_resource
def load_model_from_s3():
    s3 = boto3.client(
        "s3",
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        endpoint_url=endpoint,
        config=Config(signature_version="s3v4"),
    )
    obj = s3.get_object(Bucket=BUCKET_NAME, Key=MY_MODEL_PATH_AWS)

    model_path = MY_MODEL_TITLE_AWS
    with open(model_path, "wb") as f:
        f.write(obj["Body"].read())

    model_loaded = load_model(model_path)  # pycaret sam doda .pkl
    # st.write(f"Załadowany model: {type(model_loaded)}")   #DO DEBUGOWANIA
    return model_loaded

model = load_model_from_s3()

# ---------- 3. MAPOWANIE KOLUMN ------------------------------
COLUMN_MAP: Dict[str, str] = {
    "gdp_per_capita":        "GDP per cap",
    "social_support":        "Social Support",
    "life_expectancy":       "Healthy Life Expectence",
    "freedom":               "Life Choices Freedom",
    "generosity":            "Generosity",
    "corruption_perception": "Corruption Perception",
    "government_trust":      "Gov Belief",
}

#
# MAIN
#

# ---------- 4. INTERFEJS -------------------------------------
st.markdown(
    """
    <h2 style='text-align: center;'>🐯 Przewidywacz poziomu szczęścia</h2>
    """,
    unsafe_allow_html=True
)

with st.expander("ℹ️ Jak korzystać?", expanded=False):
    st.markdown(
        """
        1. Wybierz kraj (region ustawi się automatycznie). 
        2. Ustaw wartości wszystkich wskaźników za pomocą suwaków, dzięki czemu estymacja będzie bardziej precyzyjna.  
        3. Kliknij **Oblicz poziom szczęścia** i zobacz prognozowany wynik!⭐
        
        ⭐ Prognoza jest oparta na danych historycznych rankingu 'World Happiness Record' 
        oraz obecnej sytuacji polityczno-ekonomicznej danego kraju. 
        """,
        unsafe_allow_html=True,
    )

# ---------- 5a. KRAJ (ENG) → REGION (ENG) ---------------------
COUNTRY_TO_REGION = {
    # --- South Asia ---
    "Afghanistan": "South Asia", "Bangladesh": "South Asia", "Bhutan": "South Asia",
    "India": "South Asia", "Nepal": "South Asia", "Pakistan": "South Asia",
    "Sri Lanka": "South Asia",

    # --- Central and Eastern Europe ---
    "Albania": "Central and Eastern Europe", "Bosnia and Herzegovina": "Central and Eastern Europe",
    "Bulgaria": "Central and Eastern Europe", "Croatia": "Central and Eastern Europe",
    "Czechia": "Central and Eastern Europe", "Hungary": "Central and Eastern Europe",
    "Kosovo": "Central and Eastern Europe", "Montenegro": "Central and Eastern Europe",
    "North Macedonia": "Central and Eastern Europe", "Poland": "Central and Eastern Europe",
    "Romania": "Central and Eastern Europe", "Serbia": "Central and Eastern Europe",
    "Slovakia": "Central and Eastern Europe", "Slovenia": "Central and Eastern Europe",
    "Estonia": "Central and Eastern Europe", "Latvia": "Central and Eastern Europe",
    "Lithuania": "Central and Eastern Europe",

    # --- Middle East and North Africa ---
    "Algeria": "Middle East and North Africa", "Bahrain": "Middle East and North Africa",
    "Egypt": "Middle East and North Africa", "Iran": "Middle East and North Africa",
    "Iraq": "Middle East and North Africa", "Jordan": "Middle East and North Africa",
    "Kuwait": "Middle East and North Africa", "Lebanon": "Middle East and North Africa",
    "Libya": "Middle East and North Africa", "Morocco": "Middle East and North Africa",
    "Oman": "Middle East and North Africa", "Qatar": "Middle East and North Africa",
    "Saudi Arabia": "Middle East and North Africa", "State of Palestine": "Middle East and North Africa",
    "Syria": "Middle East and North Africa", "Tunisia": "Middle East and North Africa",
    "Turkey": "Middle East and North Africa", "United Arab Emirates": "Middle East and North Africa",
    "Yemen": "Middle East and North Africa",

    # --- Sub-Saharan Africa ---
    "Angola": "Sub-Saharan Africa", "Benin": "Sub-Saharan Africa", "Botswana": "Sub-Saharan Africa",
    "Burkina Faso": "Sub-Saharan Africa", "Burundi": "Sub-Saharan Africa",
    "Cameroon": "Sub-Saharan Africa", "Central African Republic": "Sub-Saharan Africa",
    "Chad": "Sub-Saharan Africa", "Comoros": "Sub-Saharan Africa",
    "Congo (Brazzaville)": "Sub-Saharan Africa", "Congo (Kinshasa)": "Sub-Saharan Africa",
    "Djibouti": "Sub-Saharan Africa", "Eswatini": "Sub-Saharan Africa",
    "Ethiopia": "Sub-Saharan Africa", "Gabon": "Sub-Saharan Africa", "Gambia": "Sub-Saharan Africa",
    "Ghana": "Sub-Saharan Africa", "Guinea": "Sub-Saharan Africa", "Ivory Coast": "Sub-Saharan Africa",
    "Kenya": "Sub-Saharan Africa", "Lesotho": "Sub-Saharan Africa", "Liberia": "Sub-Saharan Africa",
    "Madagascar": "Sub-Saharan Africa", "Malawi": "Sub-Saharan Africa",
    "Mali": "Sub-Saharan Africa", "Mauritania": "Sub-Saharan Africa", "Mauritius": "Sub-Saharan Africa",
    "Mozambique": "Sub-Saharan Africa", "Namibia": "Sub-Saharan Africa", "Niger": "Sub-Saharan Africa",
    "Nigeria": "Sub-Saharan Africa", "Rwanda": "Sub-Saharan Africa", "Senegal": "Sub-Saharan Africa",
    "Sierra Leone": "Sub-Saharan Africa", "Somalia": "Sub-Saharan Africa",
    "Somaliland region": "Sub-Saharan Africa", "South Africa": "Sub-Saharan Africa",
    "South Sudan": "Sub-Saharan Africa", "Sudan": "Sub-Saharan Africa",
    "Tanzania": "Sub-Saharan Africa", "Togo": "Sub-Saharan Africa", "Uganda": "Sub-Saharan Africa",
    "Zambia": "Sub-Saharan Africa", "Zimbabwe": "Sub-Saharan Africa",

    # --- Latin America and Caribbean ---
    "Argentina": "Latin America and Caribbean", "Belize": "Latin America and Caribbean",
    "Bolivia": "Latin America and Caribbean", "Brazil": "Latin America and Caribbean",
    "Chile": "Latin America and Caribbean", "Colombia": "Latin America and Caribbean",
    "Costa Rica": "Latin America and Caribbean", "Cuba": "Latin America and Caribbean",
    "Dominican Republic": "Latin America and Caribbean", "Ecuador": "Latin America and Caribbean",
    "El Salvador": "Latin America and Caribbean", "Guatemala": "Latin America and Caribbean",
    "Guyana": "Latin America and Caribbean", "Haiti": "Latin America and Caribbean",
    "Honduras": "Latin America and Caribbean", "Jamaica": "Latin America and Caribbean",
    "Mexico": "Latin America and Caribbean", "Nicaragua": "Latin America and Caribbean",
    "Panama": "Latin America and Caribbean", "Paraguay": "Latin America and Caribbean",
    "Peru": "Latin America and Caribbean", "Suriname": "Latin America and Caribbean",
    "Trinidad and Tobago": "Latin America and Caribbean", "Uruguay": "Latin America and Caribbean",
    "Venezuela": "Latin America and Caribbean",

    # --- Commonwealth of Independent States ---
    "Armenia": "Commonwealth of Independent States", "Azerbaijan": "Commonwealth of Independent States",
    "Belarus": "Commonwealth of Independent States", "Kazakhstan": "Commonwealth of Independent States",
    "Kyrgyzstan": "Commonwealth of Independent States", "Moldova": "Commonwealth of Independent States",
    "Russia": "Commonwealth of Independent States", "Tajikistan": "Commonwealth of Independent States",
    "Turkmenistan": "Commonwealth of Independent States", "Ukraine": "Commonwealth of Independent States",
    "Uzbekistan": "Commonwealth of Independent States",

    # --- North America and ANZ ---
    "United States": "North America and ANZ", "Canada": "North America and ANZ",
    "Australia": "North America and ANZ", "New Zealand": "North America and ANZ",

    # --- Western Europe ---
    "Austria": "Western Europe", "Belgium": "Western Europe", "Cyprus": "Western Europe",
    "Denmark": "Western Europe", "Finland": "Western Europe", "France": "Western Europe",
    "Germany": "Western Europe", "Greece": "Western Europe", "Iceland": "Western Europe",
    "Ireland": "Western Europe", "Italy": "Western Europe", "Luxembourg": "Western Europe",
    "Malta": "Western Europe", "Netherlands": "Western Europe", "Norway": "Western Europe",
    "Portugal": "Western Europe", "Spain": "Western Europe", "Sweden": "Western Europe",
    "Switzerland": "Western Europe", "United Kingdom": "Western Europe",

    # --- Southeast Asia ---
    "Cambodia": "Southeast Asia", "Indonesia": "Southeast Asia", "Laos": "Southeast Asia",
    "Malaysia": "Southeast Asia", "Myanmar": "Southeast Asia", "Philippines": "Southeast Asia",
    "Thailand": "Southeast Asia", "Vietnam": "Southeast Asia", "Singapore": "Southeast Asia",

    # --- East Asia ---
    "China": "East Asia", "Hong Kong S.A.R. of China": "East Asia",
    "Japan": "East Asia", "South Korea": "East Asia", "Taiwan Province of China": "East Asia",
    "Mongolia": "East Asia",
}

# ---------- 5b. TŁUMACZENIA KRAJÓW & REGIONÓW ----------------
# (słowniki COUNTRY_EN_TO_PL, COUNTRY_PL_TO_EN, REGION_EN_TO_PL, REGION_PL_TO_EN pozostają bez zmian)

COUNTRY_EN_TO_PL = {
    # --- Azja Południowa ---
    "Afghanistan": "Afganistan", "Bangladesh": "Bangladesz", "Bhutan": "Bhutan",
    "India": "Indie", "Nepal": "Nepal", "Pakistan": "Pakistan", "Sri Lanka": "Sri Lanka",

    # --- Europa Środkowo-Wschodnia ---
    "Albania": "Albania", "Bosnia and Herzegovina": "Bośnia i Hercegowina",
    "Bulgaria": "Bułgaria", "Croatia": "Chorwacja", "Czechia": "Czechy", "Hungary": "Węgry",
    "Kosovo": "Kosowo", "Montenegro": "Czarnogóra", "North Macedonia": "Macedonia Północna",
    "Poland": "Polska", "Romania": "Rumunia", "Serbia": "Serbia", "Slovakia": "Słowacja",
    "Slovenia": "Słowenia", "Estonia": "Estonia", "Latvia": "Łotwa", "Lithuania": "Litwa",

    # --- Bliski Wschód i Afryka Północna ---
    "Algeria": "Algieria", "Bahrain": "Bahrajn", "Egypt": "Egipt", "Iran": "Iran",
    "Iraq": "Irak", "Jordan": "Jordania", "Kuwait": "Kuwejt", "Lebanon": "Liban",
    "Libya": "Libia", "Morocco": "Maroko", "Oman": "Oman", "Qatar": "Katar",
    "Saudi Arabia": "Arabia Saudyjska", "State of Palestine": "Palestyna", "Syria": "Syria",
    "Tunisia": "Tunezja", "Turkey": "Turcja", "United Arab Emirates": "Zjednoczone Emiraty Arabskie",
    "Yemen": "Jemen",

    # --- Afryka Subsaharyjska ---
    "Angola": "Angola", "Benin": "Benin", "Botswana": "Botswana", "Burkina Faso": "Burkina Faso",
    "Burundi": "Burundi", "Cameroon": "Kamerun", "Central African Republic": "Republika Środkowoafrykańska",
    "Chad": "Czad", "Comoros": "Komory", "Congo (Brazzaville)": "Republika Konga",
    "Congo (Kinshasa)": "Demokratyczna Republika Konga", "Djibouti": "Dżibuti", "Eswatini": "Eswatini",
    "Ethiopia": "Etiopia", "Gabon": "Gabon", "Gambia": "Gambia", "Ghana": "Ghana",
    "Guinea": "Gwinea", "Ivory Coast": "Wybrzeże Kości Słoniowej", "Kenya": "Kenia",
    "Lesotho": "Lesotho", "Liberia": "Liberia", "Madagascar": "Madagaskar", "Malawi": "Malawi",
    "Mali": "Mali", "Mauritania": "Mauretania", "Mauritius": "Mauritius", "Mozambique": "Mozambik",
    "Namibia": "Namibia", "Niger": "Niger", "Nigeria": "Nigeria", "Rwanda": "Rwanda",
    "Senegal": "Senegal", "Sierra Leone": "Sierra Leone", "Somalia": "Somalia",
    "Somaliland region": "Somaliland", "South Africa": "Republika Południowej Afryki",
    "South Sudan": "Sudan Południowy", "Sudan": "Sudan", "Tanzania": "Tanzania", "Togo": "Togo",
    "Uganda": "Uganda", "Zambia": "Zambia", "Zimbabwe": "Zimbabwe",

    # --- Ameryka Południowa i Środkowa ---
    "Argentina": "Argentyna", "Belize": "Belize", "Bolivia": "Boliwia", "Brazil": "Brazylia",
    "Chile": "Chile", "Colombia": "Kolumbia", "Costa Rica": "Kostaryka", "Cuba": "Kuba",
    "Dominican Republic": "Dominikana", "Ecuador": "Ekwador", "El Salvador": "Salwador",
    "Guatemala": "Gwatemala", "Guyana": "Gujana", "Haiti": "Haiti", "Honduras": "Honduras",
    "Jamaica": "Jamajka", "Mexico": "Meksyk", "Nicaragua": "Nikaragua", "Panama": "Panama",
    "Paraguay": "Paragwaj", "Peru": "Peru", "Suriname": "Surinam",
    "Trinidad and Tobago": "Trynidad i Tobago", "Uruguay": "Urugwaj", "Venezuela": "Wenezuela",

    # --- Wspólnota Niepodległych Państw (po upadku ZSRR) ---
    "Armenia": "Armenia", "Azerbaijan": "Azerbejdżan", "Belarus": "Białoruś", "Kazakhstan": "Kazachstan",
    "Kyrgyzstan": "Kirgistan", "Moldova": "Mołdawia", "Russia": "Rosja", "Tajikistan": "Tadżykistan",
    "Turkmenistan": "Turkmenistan", "Ukraine": "Ukraina", "Uzbekistan": "Uzbekistan",

    # --- USA, Kanada, Australia i Nowa Zelandia ---
    "United States": "Stany Zjednoczone", "Canada": "Kanada",
    "Australia": "Australia", "New Zealand": "Nowa Zelandia",

    # --- Europa Zachodnia ---
    "Austria": "Austria", "Belgium": "Belgia", "Cyprus": "Cypr", "Denmark": "Dania",
    "Finland": "Finlandia", "France": "Francja", "Germany": "Niemcy", "Greece": "Grecja",
    "Iceland": "Islandia", "Ireland": "Irlandia", "Italy": "Włochy", "Luxembourg": "Luksemburg",
    "Malta": "Malta", "Netherlands": "Holandia", "Norway": "Norwegia", "Portugal": "Portugalia",
    "Spain": "Hiszpania", "Sweden": "Szwecja", "Switzerland": "Szwajcaria",
    "United Kingdom": "Wielka Brytania",

    # --- Azja Poł.Wschodnia ---
    "Cambodia": "Kambodża", "Indonesia": "Indonezja", "Laos": "Laos", "Malaysia": "Malezja",
    "Myanmar": "Mjanma", "Philippines": "Filipiny", "Thailand": "Tajlandia", "Vietnam": "Wietnam",
    "Singapore": "Singapur",

    # --- Azja Wschodnia ---
    "China": "Chiny", "Hong Kong S.A.R. of China": "Hongkong", "Japan": "Japonia",
    "South Korea": "Korea Południowa", "Taiwan Province of China": "Tajwan", "Mongolia": "Mongolia",
}
COUNTRY_PL_TO_EN = {pl: en for en, pl in COUNTRY_EN_TO_PL.items()}

REGION_EN_TO_PL = {
    "South Asia": "Azja Południowa",
    "Central and Eastern Europe": "Europa Środkowo‑Wschodnia",
    "Middle East and North Africa": "Bliski Wschód i Afryka Północna",
    "Sub-Saharan Africa": "Afryka Subsaharyjska",
    "Latin America and Caribbean": "Ameryka Łacińska i Karaiby",
    "Commonwealth of Independent States": "Wspólnota Niepodległych Państw",
    "North America and ANZ": "Ameryka Północna, Kanada, Australia i Nowa Zelandia",
    "Western Europe": "Europa Zachodnia",
    "Southeast Asia": "Azja Południowo‑Wschodnia",
    "East Asia": "Azja Wschodnia",
}
REGION_PL_TO_EN = {pl: en for en, pl in REGION_EN_TO_PL.items()}

# ---------- 6. LISTY DO SELECTBOXÓW --------------------------
COUNTRIES_PL = sorted(COUNTRY_PL_TO_EN.keys())
REGIONS_PL   = sorted(REGION_PL_TO_EN.keys())
YEAR         = 2025

# ---------- 7. WYBÓR KRAJU (+AUTOMATYCZNY REGION) ----------
default_country_pl = "Finlandia"
try:
    default_index = COUNTRIES_PL.index(default_country_pl)
except ValueError:
    default_index = 0

selected_country_pl = st.selectbox(
    "Wybierz kraj", COUNTRIES_PL, index=default_index
)

selected_country_en = COUNTRY_PL_TO_EN[selected_country_pl]          # do modelu
auto_region_en      = COUNTRY_TO_REGION[selected_country_en]         # region ENG
auto_region_pl      = REGION_EN_TO_PL[auto_region_en]                # na ekran

try:
    region_index = REGIONS_PL.index(auto_region_pl)
except ValueError:
    region_index = 0

st.selectbox(
    "Region (ustawiany automatycznie)",
    REGIONS_PL,
    index=region_index,
    disabled=True,
)

# ---------- 8. SUWAKI ----------------------------------------
st.markdown("### 🔧 <div style='text-align:center;'>Ustaw wskaźniki jakości życia</div>", unsafe_allow_html=True)

poziomy = ["Bardzo nisko", "Nisko", "Umiarkowanie", "Wysoko", "Bardzo wysoko"]

def opis_na_wartosc(opis, min_val, max_val):
    idx = poziomy.index(opis)
    return min_val + idx * (max_val - min_val) / 4

def tytul(nazwa):
    st.markdown(f"<div style='text-align:center; color:white; font-weight:600'>{nazwa}</div>", unsafe_allow_html=True)

# --- PKB per capita
tytul("PKB per capita")
gdp_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="gdp")
gdp_per_capita = opis_na_wartosc(gdp_opis, 5.526723, 11.663788)

# --- Wsparcie socjalne
tytul("Wsparcie socjalne")
social_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="social")
social_support = opis_na_wartosc(social_opis, 0.228217, 0.987343)

# --- Długość życia
tytul("Oczekiwana długość życia")
life_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="life")
life_expectancy = opis_na_wartosc(life_opis, 46.72, 74.474998)

# --- Wolność
tytul("Wolność wyboru")
freedom_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="freedom")
freedom = opis_na_wartosc(freedom_opis, 0.0672, 0.985178)

# --- Hojność
tytul("Hojność")
generosity_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="generosity")
generosity = opis_na_wartosc(generosity_opis, -0.337527, 0.702708)

# --- Percepcja korupcji
tytul("Postrzeganie korupcji jako powszechnego zjawiska (im niżej tym lepiej)")
corruption_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="corruption")
corruption_perception = opis_na_wartosc(corruption_opis, 0.035198, 0.983276)

# --- Zaufanie do rządu
tytul("Zaufanie do rządu")
govtrust_opis = st.select_slider("", options=poziomy, value="Umiarkowanie", key="govtrust")
government_trust = opis_na_wartosc(govtrust_opis, 0.068769, 0.993604)

# ---------- 9. DANE DO MODELU -------------------------------
def build_model_df() -> pd.DataFrame:
    return pd.DataFrame([{
        COLUMN_MAP["gdp_per_capita"]:        gdp_per_capita,
        COLUMN_MAP["social_support"]:        social_support,
        COLUMN_MAP["life_expectancy"]:       life_expectancy,
        COLUMN_MAP["freedom"]:               freedom,
        COLUMN_MAP["generosity"]:            generosity,
        COLUMN_MAP["corruption_perception"]: corruption_perception,
        COLUMN_MAP["government_trust"]:      government_trust,
        "Country": selected_country_en,      # angielska nazwa
        "Region":  auto_region_en,           # angielska nazwa
        "Year":    YEAR
    }])

def trim_to_last_full_sentence(text: str) -> str:
    """
    Przytnie tekst do ostatniej pełnej kropki (.), aby zakończyć na pełnym zdaniu.
    Jeśli kropka nie występuje, zwróci cały tekst.
    """
    last_dot_index = text.rfind('.')
    if last_dot_index == -1:
        return text.strip()
    else:
        return text[:last_dot_index+1].strip()

# ---------- 10. PREDYKCJA ------------------------------------
if st.button("Oblicz poziom szczęścia"):
    df_model = build_model_df()

    try:
        pred_df = predict_model(model, data=df_model)
        if "prediction_label" in pred_df.columns:
            pred = pred_df["prediction_label"].iloc[0]

            # Automatyczne generowanie opisu bieżącej sytuacji przez OpenAI (źródło: Wikipedia)
            try:
                def get_wiki_intro_pl(country_pl: str, chars: int = 1500) -> Optional[str]:
                    """
                    Zwraca 1-2 pierwsze akapity artykułu w polskiej Wikipedii
                    przycięte do ~chars znaków. Gdy brak artykułu - None.
                    """
                    url = "https://pl.wikipedia.org/w/api.php"
                    params = {
                        "action": "query",
                        "prop": "extracts",
                        "exintro": True,
                        "explaintext": True,
                        "format": "json",
                        "titles": country_pl,
                    }
                    resp = requests.get(url, params=params, timeout=10).json()
                    pages = resp["query"]["pages"]
                    extract = next(iter(pages.values())).get("extract")
                    if not extract:
                        return None
                    # usuwamy nagłówki sekcji i ograniczamy długość
                    clean = "\n".join(
                        line for line in extract.splitlines() if not line.startswith("==")
                    ).strip()
                    return textwrap.shorten(clean, width=chars, placeholder="…")

                # ---------------------------------------------
                country_pl = selected_country_pl               # z Twojego selectboxa
                wiki_snippet = get_wiki_intro_pl(country_pl)

                if wiki_snippet is None:
                    st.warning("Nie znaleziono artykułu w polskiej Wikipedii - używam starego promptu.")
                    generation_prompt = (
                        f"Napisz w maksymalnie 2 zdaniach po polsku aktualną "
                        f"(połowa roku 2025) sytuację polityczną i gospodarczą w kraju {country_pl}."
                    )
                    user_messages = [{"role": "user", "content": generation_prompt}]
                else:
                    user_messages = [
                        {
                            "role": "user",
                            "content": (
                                f"Poniżej masz fragment aktualnego artykułu z polskiej Wikipedii "
                                f"o kraju {country_pl} (stan połowa 2025 r.). "
                                f"Na jego podstawie podsumuj w **maksymalnie 2 zdaniach** "
                                f"obecną sytuację polityczną i gospodarczą.\n\n"
                                f"{wiki_snippet}"
                            ),
                        }
                    ]

                response = openai_client.chat.completions.create(
                    model=OPENAI_MODEL,
                    messages=[{"role": "system", "content": "Jesteś analitykiem ekonomicznym."}] + user_messages,
                    max_tokens=200,
                    temperature=0.7,
                )
                description_polish = response.choices[0].message.content.strip()
                st.info(f"📰 Aktualna sytuacja w państwie {selected_country_pl}:  \n\n{description_polish}")
            except Exception as e:
                description_polish = ""
                st.warning(f"⚠️ Nie udało się wygenerować opisu: {e}")

            # Ocena wygenerowanego opisu i korekta predykcji
            adjusted_pred = pred
            try:
                if description_polish:
                    response = openai_client.chat.completions.create(
                        model=OPENAI_MODEL,
                        messages=[
                            {"role": "system", "content": (
                                "Klasyfikuj opis sytuacji kraju jako dokładnie jedno z dwóch słów: "
                                "'dobra' (pozytywna) lub 'zla' (negatywna). "
                                "Odpowiedz tylko jednym słowem, bez dodatkowych znaków, spacji czy odmian."
                            )},
                            {"role": "user", "content": description_polish}
                        ],
                        max_tokens=3,
                        temperature=0
                    )
                    sentiment = response.choices[0].message.content.strip().lower()

                    # Prosta normalizacja i walidacja
                    if "dobra" in sentiment:
                        adjusted_pred = pred + 1
                    elif "zla" in sentiment or "zła" in sentiment:
                        adjusted_pred = pred - 1
                    else:
                        st.warning(f"⚠️ Nieoczekiwany wynik sentymentu: '{sentiment}' — nie zmieniam predykcji.")
            except Exception as e:
                st.warning(f"⚠️ Nie udało się ocenić opisu: {e}")

            pred = adjusted_pred

            st.success(f"🎉 Prognozowany poziom szczęścia na rok {YEAR}")
            st.header(f"{pred:.3f}")
        else:
            st.error("❌ Nie znaleziono kolumny z predykcją.")
    except Exception as e:
        st.error(f"❌ Błąd podczas predykcji: {e}")

    if pred < 3:
        emoji = "😢"
        desc = "Bardzo niski"
    elif pred < 6:
        emoji = "😐"
        desc = "Umiarkowany"
    elif pred < 8:
        emoji = "😊"
        desc = "Wysoki"
    else:
        emoji = "😁"
        desc = "Bardzo wysoki"

    # Wyświetlenie buźki i opisu
    st.markdown(f"### {emoji} {desc}")

    # --- TABELKA INTERPRETACYJNA -------------------------------
    happiness_scale = pd.DataFrame({
        "Zakres wyniku": ["1-3", "3-6", "6-8", "8-10"],
        "Poziom szczęścia": [
            "Bardzo niski - poważne problemy społeczne i niepewność",
            "Umiarkowany - przeciętny poziom dobrobytu",
            "Wysoki - dobre warunki do życia",
            "Bardzo wysoki - społeczeństwo bardzo zadowolone"
        ]
    })

    st.markdown("#### 🧭 Interpretacja wyniku")
    happiness_scale.index = [""] * len(happiness_scale)
    st.table(happiness_scale)

    # KONIEC 


