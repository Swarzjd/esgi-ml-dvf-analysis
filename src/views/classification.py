import joblib
import pandas as pd
import streamlit as st

from map_picker import MODELS_DIR, is_in_training_area, location_picker

MODEL_PATH = MODELS_DIR / "tree_model.joblib"


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
    latitude, longitude = location_picker("classification")

with col_form:
    st.write(f"**Latitude :** {latitude:.5f}  \n**Longitude :** {longitude:.5f}")
    if not is_in_training_area(latitude, longitude):
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
