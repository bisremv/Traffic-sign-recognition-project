import os
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Load labels
labels = pd.read_csv('labels.csv')
print(labels.head())

# Preprocess the Images
IMAGE_SIZE = 32  # Resize all images to 32x32
DATASET_PATH = './dataset'

X = []  # Features
y = []  # Labels

for index, row in labels.iterrows():
    class_folder = os.path.join(DATASET_PATH, str(row['ClassId']))
    if os.path.exists(class_folder):
        for file_name in os.listdir(class_folder):  # Iterate over all files in the folder
            img_path = os.path.join(class_folder, file_name)
            image = cv2.imread(img_path)
            if image is not None:  # Ensure image is read successfully
                image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
                X.append(image)
                y.append(row['ClassId'])
    else:
        print(f"Class folder not found: {class_folder}")

# Normalize the dataset
X = np.array(X) / 255.0  # Normalize pixel values to [0, 1]
y = np.array(y)
y = to_categorical(y, num_classes=43)  # Assuming 43 classes (0 to 42)

# Split into train, validation, and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define the model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(43, activation='softmax')  # 43 classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=15,
    batch_size=32
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc * 100:.2f}%")

# Plot the training history
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy')
plt.show()

# Save the trained model
model.save('traffic_sign_recognition_model.h5')
