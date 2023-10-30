import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk, ImageOps
import numpy as np
import matplotlib.pyplot as plt
import csv

import tensorflow.keras

# Load the model
model = tensorflow.keras.models.load_model('./tensorflow_Lesson/keras_model.h5', compile=False)

# Load the labels
with open('./tensorflow_Lesson/labels.txt', 'r') as f:
    class_name = f.read().split('\n')

# Create the array of the right shape to feed into the keras model
shape = (1, 224, 224, 3)
data = np.ndarray(shape, dtype=np.float32)

def predict_image():
    # Get the path of the selected image
    image_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])

    # Check if an image is selected
    if image_path:
        # Open the image
        image = Image.open(image_path)

        # Resize the image to 224x224 with the same strategy as in TM2
        size = (224, 224)
        image = ImageOps.fit(image, size, Image.LANCZOS)

        # Turn the image into a numpy array
        image_array = np.asarray(image)

        # Normalize the image
        normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

        # Load the image into the array
        data[0] = normalized_image_array

        # Run the inference
        predictions = model.predict(data)
        top_prediction_index = np.argmax(predictions)
        top_prediction_value = predictions[0, top_prediction_index]
        top_class_name = class_name[top_prediction_index]

        # Update the image on the GUI
        image_label.configure(image=ImageTk.PhotoImage(image))
        image_label.image = ImageTk.PhotoImage(image)

        # Update the predicted class label on the GUI
        predicted_class_label.configure(text=f"Predicted class: {top_class_name} (confidence: {top_prediction_value:.2f})")

        # Save the prediction to a CSV file
        save_prediction(image_path, top_class_name, top_prediction_value)

        # Plot the prediction probabilities
        plot_prediction_probabilities(predictions)

def save_prediction(image_path, predicted_class, confidence):
    # Append the prediction to the CSV file
    with open('predictions.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([image_path, predicted_class, confidence])

def plot_prediction_probabilities(predictions):
    # Get the class labels and probabilities
    labels = class_name[:len(predictions[0])]
    probabilities = predictions[0][:len(class_name)]
def clear_prediction():
    image_label.configure(image=None)
    predicted_class_label.def predict_image():
    # ... (previous code)
def predict_image():
    # ... (previous code)

    # Get the top N predicted classes and their confidences
    top_n = 5  # Adjust N as needed
    top_n_indices = np.argpartition(predictions, -top_n)[-top_n:]
    top_n_indices = top_n_indices[np.argsort(predictions[0, top_n_indices])[::-1]]
    
    top_classes = [class_name[i] for i in top_n_indices]
    top_confidences = [predictions[0, i] for i in top_n_indices]

    # Update the predicted class label on the GUI to show the top N predictions
    predicted_class_label.configure(text=f"Top {top_n} Predictions:\n")
    for i in range(top_n):
        predicted_class_label.configure(text=f"{top_classes[i]} (confidence: {top_confidences[i]:.2f})\n",
                                        anchor="w")

    try:
        # Run the inference
        predictions = model.predict(data)
        # ... (rest of the code)
    except Exception as e:
        predicted_class_label.configure(text=f"Prediction Error: {str(e)}")
(text="")
    

# Create a button to clear the previous prediction
clear_button = tk.Button(root, text="Clear Prediction", command=clear_prediction)
clear_button.pack(pady=10)

    # Plot the probabilities
    plt.figure(figsize=(8, 6))
    plt.bar(labels, probabilities)
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title('Prediction Probabilities')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot
    plt.show()

# Create the GUI
root = tk.Tk()
root.title("Image Classifier")
root.geometry("500x500")

# Create a button to select an image
select_image_button = tk.Button(root, text="Select Image", command=predict_image)
select_image_button.pack(pady=10)

# Create a label to display the selected image
image_label = tk.Label(root)
image_label.pack()

# Create a label to display the predicted class
predicted_class_label = tk.Label(root, text="")
predicted_class_label.pack(pady=10)

# Run the GUI main loop
root.mainloop()
