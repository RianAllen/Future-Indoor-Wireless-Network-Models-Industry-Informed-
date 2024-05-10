import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
import cv2



def preprocess_image(image_path):
    # Read the image
    image = cv2.imread(image_path)
    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to a fixed size
    resized_image = cv2.resize(gray_image, (25, 25))
    # Normalize the pixel values to be between 0 and 1
    normalized_image = resized_image / 255.0
    return normalized_image



# Load datasets
X_train = np.loadtxt('training_set.npy')
X_val = np.loadtxt('validation_set.npy')
X_test = np.loadtxt('EOLAS_test_set_real.npy')

CNN_train = np.genfromtxt('CNNtraining_set.npy', dtype=None, encoding=None)
CNN_val = np.genfromtxt('CNNvalidation_set.npy', dtype=None, encoding=None)
CNN_test = np.genfromtxt('CNNDataEOLAS_TESTSET.npy', dtype=None, encoding=None)

# Separate XY coordinates, signal strengths, and floor plan images
Y_train = X_train[:, 2]
Y_val = X_val[:, 2]
Y_test = X_test[:, 2]

BxBy_train = X_train[:, 3:5]
BxBy_val = X_val[:, 3:5]
BxBy_test = X_test[:, 3:5]

X_train_xy = X_train[:, :2]
X_val_xy = X_val[:, :2]
X_test_xy = X_test[:, :2]

# Normalize XY coordinates
X_mean, X_std = X_train_xy.mean(axis=0), X_train_xy.std(axis=0)
X_train_xy_normalized = (X_train_xy - X_mean) / X_std
X_val_xy_normalized = (X_val_xy - X_mean) / X_std
X_test_xy_normalized = (X_test_xy - X_mean) / X_std

# Normalize BxBy coordinates
BxBy_mean, BxBy_std = BxBy_train.mean(axis=0), BxBy_train.std(axis=0)
BxBy_train_normalized = (BxBy_train - BxBy_mean) / BxBy_std
BxBy_val_normalized = (BxBy_val - BxBy_mean) / BxBy_std
BxBy_test_normalized = (BxBy_test - BxBy_mean) / BxBy_std


# Combine normalized XY coordinates with additional input variables
X_train_combined = np.concatenate((X_train_xy_normalized, BxBy_train_normalized), axis=1)
X_val_combined = np.concatenate((X_val_xy_normalized, BxBy_val_normalized), axis=1)
X_test_combined = np.concatenate((X_test_xy_normalized, BxBy_test_normalized), axis=1)

# Preprocess floor plan images in the training set
X_train_images_normalized = []
for image_path in CNN_train:  
    preprocessed_image = preprocess_image(image_path)
    X_train_images_normalized.append(preprocessed_image)
X_train_images_normalized = np.array(X_train_images_normalized)

# Preprocess floor plan images in the validation set
X_val_images_normalized = []
for image_path in CNN_val:  
    preprocessed_image = preprocess_image(image_path)
    X_val_images_normalized.append(preprocessed_image)
X_val_images_normalized = np.array(X_val_images_normalized)

# Preprocess floor plan images in the test set
X_test_images_normalized = []
for image_path in CNN_test:  
    preprocessed_image = preprocess_image(image_path)
    X_test_images_normalized.append(preprocessed_image)
X_test_images_normalized = np.array(X_test_images_normalized)

# Determining the height and width from the preprocessed image
height, width = X_train_images_normalized[0].shape 
# Grayscale image has 1 channel
channels = 1 
# Defining the CNN structure for processing images
image_input = keras.Input(shape=(height, width, channels), name='image_input')
x = layers.Conv2D(32, kernel_size=(3, 3), activation='relu')(image_input)
x = layers.MaxPooling2D(pool_size=(2, 2))(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
image_output = layers.Dense(32, activation='relu')(x)

# Defining the ANN structure for processing XY coordinates
xyBxBy_input = keras.Input(shape=(4,), name='xyBxBy_input')
xy_output = layers.Dense(256, activation='relu')(xyBxBy_input)
xy_output = layers.Dense(128, activation='relu')(xy_output)
xy_output = layers.Dense(64, activation='relu')(xy_output)
xy_output = layers.Dense(32, activation='relu')(xy_output)
# Concatenating outputs from both structures
concatenated = layers.Concatenate()([image_output, xy_output])
# Final output layer
output = layers.Dense(1, name='output')(concatenated)
# Define the combined model
model = keras.Model(inputs=[image_input, xyBxBy_input], outputs=output)

# Implementing learning rate scheduling
initial_learning_rate = 0.001
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Compiling the model
model.compile(optimizer=optimizer, loss='mse')

# Training the model with validation data
hist = model.fit({'image_input': X_train_images_normalized, 'xyBxBy_input': X_train_combined},
                 {'output': Y_train},
                 validation_data=({'image_input': X_val_images_normalized, 'xyBxBy_input': X_val_combined}, {'output': Y_val}),
                 epochs=10, batch_size=64, verbose=1)

# Evaluating the model on the test set
test_loss = model.evaluate({'image_input': X_test_images_normalized, 'xyBxBy_input': X_test_combined}, {'output': Y_test})
print("Test Loss:", test_loss)

# Save the predicted values to the NumPy file
# Predict on the test set
Y_pred = model.predict({'image_input': X_test_images_normalized, 'xyBxBy_input': X_test_combined})

# Since Y_pred is an array, you directly access its elements
output = Y_pred[:, 0] 

# Save the predicted values to the NumPy file
output_file_path = 'predicted_strengths.npy'
np.savetxt(output_file_path, output, fmt='%d')

# Plotting Training Vs Validation Loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc='upper left')
plt.show()