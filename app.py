import tkinter as tk
from tkinter import filedialog, Label, Button, Frame
import cv2
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image, ImageTk

# Load the trained model
model = load_model('traffic_sign_recognition_model.h5')

# Class names from labels.csv
classes = [
    "Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)",
    "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)",
    "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)",
    "No passing", "No passing for vehicles over 3.5 metric tons",
    "Right-of-way at the next intersection", "Priority road", "Yield",
    "Stop", "No vehicles", "Vehicles over 3.5 metric tons prohibited",
    "No entry", "General caution", "Dangerous curve to the left",
    "Dangerous curve to the right", "Double curve", "Bumpy road",
    "Slippery road", "Road narrows on the right", "Road work",
    "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed and passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left", "Keep right",
    "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing by vehicles over 3.5 metric tons"
]

def recognize_image(img_path):
    # Load and preprocess the image
    image = cv2.imread(img_path)
    image = cv2.resize(image, (32, 32))  # Resize to match model input
    image = np.expand_dims(image, axis=0) / 255.0  # Normalize

    # Predict
    predictions = model.predict(image)
    class_id = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    return classes[class_id], confidence

def open_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if file_path:
        label_path.config(text=f"File: {file_path}", fg="#00FF00")
        recognized_class, confidence = recognize_image(file_path)
        label_result.config(
            text=f"Predicted: {recognized_class} \nConfidence: {confidence:.2f}%", fg="#00FF00"
        )

        # Display the selected image
        img = Image.open(file_path)
        img = img.resize((300, 300))
        img_tk = ImageTk.PhotoImage(img)
        label_image.config(image=img_tk)
        label_image.image = img_tk

# Create the Tkinter application
app = tk.Tk()
app.title("🚦 Traffic Sign Recognition")
app.configure(bg="#121212")  # Dark theme

# Configure grid layout for responsiveness
app.rowconfigure(0, weight=1)
app.rowconfigure(1, weight=1)
app.columnconfigure(0, weight=1)

# Add a glowing frame for futuristic style
frame_border = Frame(app, bg="#00FF00", bd=5)
frame_border.grid(row=0, column=0, pady=20, padx=20, sticky="nsew")

main_frame = Frame(frame_border, bg="#222222", relief="flat", bd=0)
main_frame.pack(fill="both", expand=True)

# Header Section
header_label = Label(
    main_frame, 
    text="🚦 Traffic Sign Recognition 🚦", 
    font=("Orbitron", 24, "bold"), 
    bg="#222222", 
    fg="#00FF00"
)
header_label.pack(pady=20)

# Image Preview and File Selection
label_path = Label(
    main_frame, 
    text="Choose an image to recognize", 
    font=("Consolas", 12), 
    bg="#222222", 
    fg="#FFFFFF"
)
label_path.pack(pady=10)

button_browse = Button(
    main_frame, 
    text="Browse Image", 
    command=open_file, 
    font=("Consolas", 14, "bold"), 
    bg="#00FF00", 
    fg="#121212", 
    relief="flat", 
    padx=20, 
    pady=10
)
button_browse.pack(pady=10)

label_image = Label(main_frame, bg="#333333", relief="solid", bd=1, width=300, height=300)
label_image.pack(pady=20)

# Prediction Results
label_result = Label(
    main_frame, 
    text="Predicted: None", 
    font=("Orbitron", 14, "bold"), 
    bg="#222222", 
    fg="#FFFFFF", 
    wraplength=400, 
    justify="center"
)
label_result.pack(pady=10)

# Footer Branding
footer_label = Label(
    app, 
    text="© 2024 Traffic Sign Recognition developed by bisrat", 
    bg="#121212", 
    fg="#777777", 
    font=("Orbitron", 10)
)
footer_label.grid(row=1, column=0, pady=10, sticky="nsew")

# Make the app responsive
for i in range(2):
    app.rowconfigure(i, weight=1)
app.columnconfigure(0, weight=1)

# Run the app
app.mainloop()
