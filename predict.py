import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

MODEL_PATH = 'skin_model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = {
    'akiec': 'Actinic Keratoses and Intraepithelial Carcinoma (Bowen\'s disease)',
    'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis',
    'df': 'Dermatofibroma',
    'mel': 'Melanoma',
    'nv': 'Melanocytic Nevi',
    'vasc': 'Vascular Lesions'
}

def predict_skin_disease(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = list(CLASS_NAMES.keys())[np.argmax(prediction[0])]
    confidence = np.max(prediction[0]) * 100
    
    full_name = CLASS_NAMES[predicted_class]
    
    return full_name, confidence, np.array(img)

def get_valid_file_path():
    while True:
        img_path = input("\nEnter the path to the image: ").strip()
        if os.path.exists(img_path):
            return img_path
        else:
            print("❌ Error: File not found! Please try again.")

def main():
    try:
        img_path = get_valid_file_path()
        predicted_class, confidence, img = predict_skin_disease(img_path)

        # Display the image with prediction
        plt.imshow(img.astype("uint8"))
        plt.axis('off')
        plt.title(f'Prediction: {predicted_class} ({confidence:.2f}%)')
        plt.show()

        # Print the prediction
        print(f"\n🩺 **Predicted:** {predicted_class} ({confidence:.2f}%)")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()