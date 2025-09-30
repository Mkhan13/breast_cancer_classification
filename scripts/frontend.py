import streamlit as st
from PIL import Image
from .model import BreastCancerModel

MODEL_PATH = "model.pth"
model = BreastCancerModel(MODEL_PATH)

def run_frontend():
    ''' Streamlit frontend for image upload and prediction '''
    st.title("Breast Cancer Image Classifier")
    st.write("Upload an image to predict if it's benign or malignant.")

    uploaded_file = st.file_uploader("Choose an image:", type=["png", "jpg", "jpeg"])
    
    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        if st.button("Predict"):
            pred_class, prob = model.predict(img)
            st.write(f"Predicted Class: {'Malignant' if pred_class == 1 else 'Benign'}")
            st.write(f"Prediction Probability: {prob:.4f}")