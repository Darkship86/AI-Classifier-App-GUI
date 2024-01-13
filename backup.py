import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import numpy as np
import tensorflow as tf

# Load your trained image model
model = tf.keras.models.load_model('keras_model.h5')

# Function to preprocess the image
def preprocess_image(image):
    # Resize the image to match the input size of your model
    image = image.resize((224, 224))
    # Convert the image to a numpy array
    image = np.array(image)
    # Normalize the image
    image = image / 255.0
    # Expand dimensions to match the input shape expected by your model
    image = np.expand_dims(image, axis=0)
    return image

# Function to classify the image
def classify_image():
    # Open a file dialog to select an image
    file_path = filedialog.askopenfilename()
    # Load the selected image
    image = Image.open(file_path)
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Make predictions using your model
    predictions = model.predict(preprocessed_image)
    # Get the predicted class label
    predicted_class = np.argmax(predictions)
    # Display the predicted class label
    output_label.config(text=f"Predicted Class: {predicted_class}")

# Create the Tkinter app window
window = tk.Tk()
window.title("Animal Classification App By Daniel KC")



# Create a button to select an image
select_button = tk.Button(window, text="Select JPG Image", command=classify_image)
select_button.pack(pady=10)

# Create a label to display the output
output_label = tk.Label(window, text="")
output_label.pack(pady=10)

# Run the Tkinter event loop
window.mainloop()