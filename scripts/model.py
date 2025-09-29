import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image

class SimpleCNN(nn.Module):
    """Custom CNN for binary classification"""
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential( # Convolutional layers
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential( # Fully connected layers
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 1),
        )

    def forward(self, x): # Forward pass
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

class BreastCancerModel:
    """Wrapper for loading model and making predictions"""
    def __init__(self, model_path="model.pth"):
        self.model = SimpleCNN().to(device)
        self.model.load_state_dict(torch.load(model_path, map_location=device)) # Load model weights
        self.model.eval() # Set model to evaluation mode
        self.transform = transforms.Compose([  # Image transformations
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def predict(self, image: Image.Image):
        """Return predicted class (0/1) and probability"""
        img_t = self.transform(image).unsqueeze(0).to(device) # Add batch dimension
        with torch.no_grad(): # Disable gradient calculation
            output = self.model(img_t).squeeze(1) # Get model output
            prob = torch.sigmoid(output).item() # Calculate probability using sigmoid
            if prob >= 0.5:
                pred_class = 1
            else:
                pred_class = 0
        return pred_class, round(prob, 4)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Use GPU if available
    model_wrapper = BreastCancerModel("model.pth")
    img = Image.open("test_malignant2.png") # Test image
    print(model_wrapper.predict(img))