# NN Imports
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

# Image data converted to x,y coordinates with corresponding intensity values (200x200 points)
X_train = np.loadtxt('training_set.npy')
X_val = np.loadtxt('validation_set.npy')
X_test = np.loadtxt('test_set_TESTMAP.npy')

#Separating Inputs and Outputs
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


# Defining the input shape
input_shape = X_train_combined.shape[1]
# Defining the model architecture
# Varying the amount of layers/neurons for accuracy
model = keras.Sequential([
    layers.Dense(256, activation='relu', input_shape=(input_shape,)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu'),
    layers.Dense(1)
])

# Implementing learning rate scheduling
initial_learning_rate = 0.005
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate, decay_steps=10000, decay_rate=0.96, staircase=True
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# Compiling the model
model.compile(optimizer=optimizer, loss='mse')

# Training the model with validation data
hist = model.fit(X_train_combined, Y_train, epochs=300, batch_size=64, 
                 validation_data=(X_val_combined, Y_val), verbose=1)

# Evaluating the model on the test set
test_loss = model.evaluate(X_test_combined, Y_test)
print("Test Loss:", test_loss)
Y_pred = model.predict(X_test_combined)

# Printing the predicted strengths
output_file_path = 'predicted_strengths_ANN.npy'

# Save the predicted values to the NumPy file
np.savetxt(output_file_path, Y_pred, fmt='%d')

#Plotting Training Vs Validation Loss
plt.figure()
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','val'], loc = 'upper left')
plt.show()