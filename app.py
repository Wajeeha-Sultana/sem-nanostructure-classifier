import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow import keras

st.set_page_config(page_title="SEM Nanostructure Classifier", page_icon="🔬", layout="centered")

CLASS_NAMES = [
    "nanowire_fibre_mesh",
    "particle_coated_surface",
    "patterned_structure",
    "porous_fibrous_network",
]

CLASS_DESCRIPTIONS = {
    "nanowire_fibre_mesh": "Dense mat of long, tangled nanowire/fibre strands.",
    "particle_coated_surface": "Fine granular/particle-like texture coating a surface.",
    "patterned_structure": "Sharp-edged, geometric lithographic/electrode-like pattern.",
    "porous_fibrous_network": "Fine fibrous mesh with visible pores and occasional larger strands.",
}

IMG_SIZE = (128, 128)


@st.cache_resource
def load_model():
    return keras.models.load_model("sem_nanostructure_cnn.keras")


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("L").resize(IMG_SIZE)
    arr = np.array(image, dtype=np.float32) / 255.0
    arr = np.expand_dims(arr, axis=(0, -1))  # (1, 128, 128, 1)
    return arr


st.title("🔬 SEM Nanostructure-Type Classifier")
st.write(
    "Upload a Scanning Electron Microscopy (SEM) image and this CNN will classify it "
    "into one of four nanostructure morphology categories: nanowire/fibre mesh, "
    "particle-coated surface, patterned/lithographic structure, or porous fibrous network."
)

with st.expander("About this model"):
    st.write(
        "Trained from scratch on 435 real SEM images sourced from the "
        "[NFFA-EUROPE annotated SEM image dataset](https://doi.org/10.1038/sdata.2018.172) "
        "(Aversa et al., 2018), via the "
        "[motiurinfo/SEM-Dataset-500](https://github.com/motiurinfo/SEM-Dataset-500) curated subset. "
        "Built as a NAVTTC AI (ML & Deep Learning) CNN capstone project, relevant to SEM-based "
        "materials/nanocomposite characterization workflows."
    )

uploaded_file = st.file_uploader("Upload an SEM image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded image", use_container_width=True)

    model = load_model()
    x = preprocess(image)
    probs = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    pred_class = CLASS_NAMES[pred_idx]

    st.subheader(f"Prediction: `{pred_class}`")
    st.write(CLASS_DESCRIPTIONS[pred_class])
    st.write(f"Confidence: **{probs[pred_idx]*100:.1f}%**")

    st.write("Full class probabilities:")
    for c, p in sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1]):
        st.progress(float(p), text=f"{c}: {p*100:.1f}%")
else:
    st.info("Upload an SEM image above to get a prediction.")

st.divider()
st.caption(
    "Note: trained on a small (435-image), imbalanced dataset — a portfolio/coursework demo, "
    "not a validated tool for production materials-characterization decisions."
)
