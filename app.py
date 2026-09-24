from pathlib import Path

import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Ghoul, Goblin, or Ghost?")

MODEL_PATH = Path(__file__).parent / "ghosts_model.pkl"
COLORS = ["black", "blood", "blue", "clear", "green", "white"]
COLS = ["bone_length", "rotting_flesh", "hair_length", "has_soul"] + [f"color_{c}" for c in COLORS]
ICONS = {"Ghoul": "🧟", "Goblin": "👹", "Ghost": "👻"}

PRESETS = {
    "Enter manually": None,
    "Typical Ghost": (0.347, 0.594, 0.373, 0.320, "white"),
    "Typical Ghoul": (0.517, 0.491, 0.656, 0.602, "white"),
    "Typical Goblin": (0.430, 0.442, 0.545, 0.479, "clear"),
}


@st.cache_resource
def load_model():
    return pickle.load(open(MODEL_PATH, "rb"))


model = load_model()

st.title("Ghoul, Goblin, or Ghost?")
st.write(
    "A Random Forest model (trained on the Kaggle 'Ghouls, Goblins, and Ghosts... Boo!' dataset, 371 monsters) "
    "predicts a monster's type from its physical measurements. Note: goblins and ghouls look alike and are "
    "often confused, even by the model (validation accuracy ≈ 77%)."
)

choice = st.selectbox("Load an example (optional)", list(PRESETS.keys()))
d = PRESETS[choice] if PRESETS[choice] else (0.45, 0.45, 0.45, 0.45, "clear")

col1, col2 = st.columns(2)
with col1:
    bone_length = st.slider("Bone length", 0.0, 1.0, d[0], 0.01)
    rotting_flesh = st.slider("Rotting flesh", 0.0, 1.0, d[1], 0.01)
with col2:
    hair_length = st.slider("Hair length", 0.0, 1.0, d[2], 0.01)
    has_soul = st.slider("Has soul", 0.0, 1.0, d[3], 0.01)
color = st.selectbox("Dominant color", COLORS, index=COLORS.index(d[4]))

if st.button("Predict"):
    row = {c: 0 for c in COLS}
    row.update(bone_length=bone_length, rotting_flesh=rotting_flesh, hair_length=hair_length, has_soul=has_soul)
    row[f"color_{color}"] = 1
    X = pd.DataFrame([row])[COLS]

    pred = model.predict(X)[0]
    proba = dict(zip(model.classes_, model.predict_proba(X)[0]))
    st.success(f"{ICONS[pred]} Predicted type: **{pred}**")
    st.bar_chart(pd.Series(proba, name="probability"))

st.caption(
    "Hair length and 'has soul' matter most for this model; color has almost no effect. "
    "Ghosts are usually easy to tell apart, but goblins and ghouls are genuinely hard to separate."
)
