import joblib
from PIL import Image
import numpy as np
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
import torch

# Initialize the FaceNet model (InceptionResnetV1)
inception_resnet = InceptionResnetV1(pretrained='vggface2').eval()

# Function to extract FaceNet features from an image
def extract_facenets_features(image, resize_dim=(160, 160)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_resized = image.resize(resize_dim)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    image_tensor = transform(image_resized).unsqueeze(0)
    
    with torch.no_grad():
        embedding = inception_resnet(image_tensor).cpu().numpy()
    
    return embedding.flatten()

# Function to load the .pkl files (model and label encoder)
def load_model_and_encoder(model_path, encoder_path):
    # Load the saved SVM classifier model
    svm_classifier = joblib.load(model_path)

    # Load the label encoder
    label_encoder = joblib.load(encoder_path)

    return svm_classifier, label_encoder

# Function to predict an image's label
def predict_image(image_path, model, label_encoder):
    # Load the image
    img = Image.open(image_path)

    # Extract features using FaceNet
    features = extract_facenets_features(img)

    # Make a prediction using the trained model
    prediction = model.predict([features])
    
    # Get the predicted label
    predicted_label = label_encoder.inverse_transform(prediction)

    print(f"Predicted label: {predicted_label[0]}")

# Example usage
if __name__ == "__main__":
    # Specify the paths to the .pkl files
    model_path = '/Users/sujalkuthe/Downloads/svm_classifier_model1.pkl'  # Replace with the actual path
    encoder_path = '/Users/sujalkuthe/Downloads/label_encoder1.pkl'  # Replace with the actual path

    # Load the model and label encoder
    svm_classifier, label_encoder = load_model_and_encoder(model_path, encoder_path)

    # Specify the image path to be predicted
    image_path = '/Users/sujalkuthe/Downloads/Brock-Lesnar-1511404.jpg'

    # Predict the image label
    predict_image(image_path, svm_classifier, label_encoder)
