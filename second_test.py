import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

# Step 1: Splitting the dataset

def load_dataset():
    X, y = [], []
    cancer_path = '/Users/aleksandarmitrovic/Downloads/RIS2023 2/rakave'
    non_cancer_path = '/Users/aleksandarmitrovic/Downloads/RIS2023 2/zdrave'
    
    # Load the cancer images
    for filename in os.listdir(cancer_path):
        img = Image.open(os.path.join(cancer_path, filename))
        X.append(np.array(img))
        y.append(1)
    
    # Load the non-cancer images
    for filename in os.listdir(non_cancer_path):
        img = Image.open(os.path.join(non_cancer_path, filename))
        X.append(np.array(img))
        y.append(0)
    
    return np.array(X), np.array(y)

X, y = load_dataset()

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.2, random_state=42)


# Step 2: Preprocessing the images

datagen = ImageDataGenerator(rotation_range=10, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)

# Step 3: Choosing a machine learning algorithm

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Step 4: Training and evaluating the model

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(datagen.flow(X_train, y_train, batch_size=32), epochs=10, validation_data=(X_val, y_val))

val_loss, val_acc = model.evaluate(X_val, y_val)

test_loss, test_acc = model.evaluate(X_test, y_test)


# Step 5: Evaluating the performance of the final model

print(f"Validation loss: {val_loss}, Validation accuracy: {val_acc}")
print(f"Test loss: {test_loss}, Test accuracy: {test_acc}")

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Step 6: Making predictions on new data

#new_img = Image.open('path/to/new_image.png')

#new_img = new_img.resize((256, 256))
#new_img_array = np.array(new_img) / 255.0

#prediction = model.predict(np.array([new_img_array]))

#print(f"Predicted probability of having breast cancer: {prediction[0][0]}")
