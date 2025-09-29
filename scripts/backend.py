from PIL import Image
from model import BreastCancerModel

MODEL_PATH = "model.pth"
model = BreastCancerModel(MODEL_PATH) # Initialize the model

def predict_image(image: Image.Image):
    ''' Run prediction using the CNN model '''
    pred_class, probability = model.predict(image)
    return {"pred_class": pred_class, "probability": probability}