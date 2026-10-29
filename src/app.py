from pathlib import Path

import folium
import joblib
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "tree_model.joblib"

# Emprise géographique des biens du jeu d'entraînement (Hauts-de-France)
LAT_MIN, LAT_MAX = 48.87, 51.08
LON_MIN, LON_MAX = 1.38, 4.21
DEFAULT_LOCATION = (50.63, 3.06)  # Lille

st.set_page_config(page_title="Classification DVF", page_icon="🏠", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


if not MODEL_PATH.exists():
    st.error(
        "Modèle introuvable : exécutez le notebook `analysis-dvf.ipynb` "
        "jusqu'à la cellule de sauvegarde (`./models/tree_model.joblib`)."
    )
    st.stop()

bundle = load_model()
tree_model, label_encoder, features = bundle["model"], bundle["label_encoder"], bundle["features"]

if "location" not in st.session_state:
    st.session_state.location = DEFAULT_LOCATION

st.title("Appartement ou maison ?")
st.write(
    "Renseignez les caractéristiques du bien et placez-le sur la carte : "
    "l'arbre de décision entraîné dans le notebook prédit son type."
)

col_form, col_map = st.columns([1, 2], gap="large")

with col_form:
    st.subheader("Caractéristiques")
    surface = st.number_input("Surface réelle bâtie (m²)", min_value=1, value=35, step=1)
    pieces = st.number_input("Nombre de pièces principales", min_value=1, value=1, step=1)
    valeur = st.number_input("Valeur foncière (€)", min_value=1, value=95000, step=5000)

with col_map:
    st.subheader("Emplacement")
    st.caption("Cliquez sur la carte pour placer le bien. Le cadre en pointillés correspond à la zone couverte par les données d'entraînement.")

    base_map = folium.Map(location=DEFAULT_LOCATION, zoom_start=9)
    folium.Rectangle(
        bounds=[(LAT_MIN, LON_MIN), (LAT_MAX, LON_MAX)],
        color="#555555",
        weight=1,
        dash_array="6",
        fill=False,
    ).add_to(base_map)

    # Le marqueur est passé à part pour ne pas recharger la carte (zoom conservé) à chaque clic
    marker_group = folium.FeatureGroup(name="Bien")
    marker_group.add_child(folium.Marker(st.session_state.location, tooltip="Bien à classer"))

    map_state = st_folium(
        base_map,
        feature_group_to_add=marker_group,
        key="map",
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    clicked = map_state.get("last_clicked") if map_state else None
    if clicked:
        new_location = (round(clicked["lat"], 5), round(clicked["lng"], 5))
        if new_location != st.session_state.location:
            st.session_state.location = new_location
            st.rerun()

latitude, longitude = st.session_state.location

with col_form:
    st.write(f"**Latitude :** {latitude:.5f}  \n**Longitude :** {longitude:.5f}")
    if not (LAT_MIN <= latitude <= LAT_MAX and LON_MIN <= longitude <= LON_MAX):
        st.warning("Ce point est hors de la zone des données d'entraînement : la prédiction est peu fiable.")

    # Mêmes noms et même ordre de colonnes que lors de l'entraînement
    new_property = pd.DataFrame([{
        "surface_reelle_bati": surface,
        "nombre_pieces_principales": pieces,
        "valeur_fonciere": valeur,
        "longitude": longitude,
        "latitude": latitude,
    }])[features]

    prediction = tree_model.predict(new_property)[0]
    probabilities = tree_model.predict_proba(new_property)[0]
    predicted_type = label_encoder.inverse_transform([prediction])[0]

    st.divider()
    st.subheader("Prédiction")
    st.metric("Type de bien", predicted_type)
    for classe, proba in zip(label_encoder.classes_, probabilities):
        st.progress(float(proba), text=f"{classe} : {proba * 100:.1f} %")
    st.caption(f"Prix au m² : {valeur / surface:,.0f} €".replace(",", " "))
