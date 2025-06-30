import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
DATASET_DIR = "leapGestRecog"
IMG_SIZE = 64
def load_images(dataset_path):
    X = []
    y = []
    for user in os.listdir(dataset_path):
        user_path = os.path.join(dataset_path, user)
        if os.path.isdir(user_path):
            for gesture in os.listdir(user_path):
                gesture_path = os.path.join(user_path, gesture)
                if os.path.isdir(gesture_path):
                    for img_file in os.listdir(gesture_path):
                        if img_file.endswith(".png"):
                            img_path = os.path.join(gesture_path, img_file)
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                            X.append(img)
                            y.append(gesture)
    return np.array(X), np.array(y)
print("Loading data...")
X, y = load_images(DATASET_DIR)
X = X / 255.0
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(len(np.unique(y_encoded)), activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("Training model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
loss, acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {acc * 100:.2f}%")
predictions = model.predict(X_test[:40])
for i, pred in enumerate(predictions):
    plt.imshow(X_test[i].reshape(IMG_SIZE, IMG_SIZE), cmap='gray')
    plt.title(f"Predicted: {le.inverse_transform([np.argmax(pred)])[0]}")
    plt.axis('off')
    plt.show()