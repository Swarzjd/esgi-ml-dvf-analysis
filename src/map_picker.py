from pathlib import Path

import folium
import streamlit as st
from streamlit_folium import st_folium

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Emprise géographique des biens du jeu d'entraînement (Hauts-de-France)
LAT_MIN, LAT_MAX = 48.87, 51.08
LON_MIN, LON_MAX = 1.38, 4.21
DEFAULT_LOCATION = (50.63, 3.06)  # Lille


def is_in_training_area(latitude, longitude):
    return LAT_MIN <= latitude <= LAT_MAX and LON_MIN <= longitude <= LON_MAX


def location_picker(key):
    """Affiche la carte, place un marqueur au clic et renvoie (latitude, longitude)."""
    state_key = f"{key}_location"
    if state_key not in st.session_state:
        st.session_state[state_key] = DEFAULT_LOCATION

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
    marker_group.add_child(folium.Marker(st.session_state[state_key], tooltip="Bien"))

    map_state = st_folium(
        base_map,
        feature_group_to_add=marker_group,
        key=f"{key}_map",
        height=520,
        use_container_width=True,
        returned_objects=["last_clicked"],
    )

    clicked = map_state.get("last_clicked") if map_state else None
    if clicked:
        new_location = (round(clicked["lat"], 5), round(clicked["lng"], 5))
        if new_location != st.session_state[state_key]:
            st.session_state[state_key] = new_location
            st.rerun()

    return st.session_state[state_key]
