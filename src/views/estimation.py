import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.neighbors import BallTree

from map_picker import MODELS_DIR, is_in_training_area, location_picker

MODEL_PATH = MODELS_DIR / "regression_model.joblib"
EARTH_RADIUS_KM = 6371
# Au-delà de cette distance de la transaction connue la plus proche, le code postal déduit est douteux
MAX_NEIGHBOUR_KM = 5


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    locations = bundle["locations"]
    tree = BallTree(np.radians(locations[["latitude", "longitude"]].to_numpy()), metric="haversine")
    return bundle, tree


def nearest_location(tree, locations, latitude, longitude):
    """Renvoie la transaction connue la plus proche du point et sa distance en km."""
    dist, idx = tree.query(np.radians([[latitude, longitude]]), k=1)
    return locations.iloc[idx[0][0]], dist[0][0] * EARTH_RADIUS_KM


if not MODEL_PATH.exists():
    st.error(
        "Modèle introuvable : exécutez le notebook `regression-analysis.ipynb` "
        "jusqu'à la cellule de sauvegarde (`./models/regression_model.joblib`)."
    )
    st.stop()

bundle, location_tree = load_model()
model, features, dep_cols, locations = bundle["model"], bundle["features"], bundle["dep_cols"], bundle["locations"]

st.title("Estimation du prix d'un bien")
st.write(
    "Renseignez les caractéristiques du bien et placez-le sur la carte : "
    f"le modèle {bundle['model_name']} entraîné dans le notebook de régression estime sa valeur foncière."
)

col_form, col_map = st.columns([1, 2], gap="large")

with col_form:
    st.subheader("Caractéristiques")
    type_local = st.selectbox("Type de bien", ["Appartement", "Maison"])
    surface = st.number_input("Surface réelle bâtie (m²)", min_value=1, value=70, step=1)
    pieces = st.number_input("Nombre de pièces principales", min_value=1, value=3, step=1)

with col_map:
    st.subheader("Emplacement")
    latitude, longitude = location_picker("estimation")

# Le modèle attend aussi le code postal et le département : on les reprend de la transaction la plus proche
neighbour, distance_km = nearest_location(location_tree, locations, latitude, longitude)
departement = next(c for c in dep_cols if neighbour[c]).removeprefix("code_departement_")
# code_postal est relu en float par pandas (valeurs manquantes possibles)
code_postal = "inconnu" if pd.isna(neighbour["code_postal"]) else f"{int(neighbour['code_postal']):05d}"

with col_form:
    st.write(
        f"**Latitude :** {latitude:.5f}  \n**Longitude :** {longitude:.5f}  \n"
        f"**Code postal :** {code_postal} (département {departement})"
    )
    if not is_in_training_area(latitude, longitude) or distance_km > MAX_NEIGHBOUR_KM:
        st.warning(
            f"Aucune vente connue à moins de {MAX_NEIGHBOUR_KM} km "
            f"(la plus proche est à {distance_km:.1f} km) : l'estimation est peu fiable."
        )

    # Mêmes noms et même ordre de colonnes que lors de l'entraînement
    new_property = pd.DataFrame([{
        "surface_reelle_bati": surface,
        "nombre_pieces_principales": pieces,
        "type_local": type_local,
        "code_postal": neighbour["code_postal"],
        "longitude": longitude,
        "latitude": latitude,
        **{c: bool(neighbour[c]) for c in dep_cols},
    }])[features]

    # Le modèle a été entraîné sur log(1 + prix) : on repasse en euros
    price = float(np.expm1(model.predict(new_property)[0]))
    mae = bundle["mae"]

    st.divider()
    st.subheader("Estimation")
    st.metric("Valeur foncière estimée", f"{price:,.0f} €".replace(",", " "))
    st.write(f"Prix au m² : **{price / surface:,.0f} €**".replace(",", " "))
    st.caption(
        f"Erreur moyenne du modèle sur le jeu de test : ± {mae:,.0f} € (MAE), "
        f"R² = {bundle['r2']:.3f}.".replace(",", " ")
    )
