

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import sys
import io

# Set the default encoding to UTF-8 to avoid UnicodeEncodeError
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Load the saved model
model = load_model('waste_classifier_model.h5')

# Optionally, recompile the model if you want to avoid the metrics warning
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Function to preprocess and predict a single image
def predict_image(img_path):
    # Load the image with the target size
    img = image.load_img(img_path, target_size=(128, 128))  # Use the same size as during training
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Rescale image pixels to [0, 1]

    # Make the prediction
    prediction = model.predict(img_array)

    # Map the prediction result
    if prediction > 0.5:
        result = 'Recycle'  # Assuming 1 is 'Recycle'
    else:
        result = 'Organic'  # Assuming 0 is 'Organic'

    return result, prediction

# Example usage
img_path = 'C:/Users/imroz/pictures/phone.jpg'  # Replace with the path to the image you want to predict
result, prediction = predict_image(img_path)

# Display the result and prediction
print(f"Prediction: {result}")
print(f"Prediction Score: {prediction[0][0]:.4f}")

# Show the image
img = image.load_img(img_path, target_size=(128, 128))
plt.imshow(img)
plt.axis('off')
plt.show()


