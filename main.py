import tkinter as tk
from tkinter import filedialog
from keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

# Load the model
model = load_model("keras_Model.h5", compile=False)
class_names = open("labels.txt", "r").readlines()

def predict_image():
    # Open file dialog to select an image
    file_path = filedialog.askopenfilename()
    
    # Load and preprocess the image
    image = Image.open(file_path).convert("RGB")
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.Resampling.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.5) - 1
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    data[0] = normalized_image_array
    
    # Make prediction
    prediction = model.predict(data)
    index = np.argmax(prediction)
    class_name = class_names[index]
    confidence_score = prediction[0][index]
    
    # Display prediction and confidence score
    result_label.config(text=f"Animal: {class_name[2:]}\nConfidence Score: {confidence_score}")

# Create the Tkinter app
app = tk.Tk()
app.title("Animal Classifier App by Daniel KC")
app.geometry("650x650")

subheading_label = tk.Label(app, text="12 different animal species", font=("Arial", 19, "bold"))
subheading_label.pack()


subheading_label = tk.Label(app, text="Try it out below !  (Note: the image should be in JPG format)", font=("Arial", 19,))
subheading_label.pack()

# Create a button to select an image
select_button = tk.Button(app, text="Select JPG Image", command=predict_image)
select_button.pack(pady=20)

# Create a label to display the result
result_label = tk.Label(app, text="")
result_label.pack()

# Run the Tkinter event loop
app.mainloop()