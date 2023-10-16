import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt


# Load and preprocess data
def load_data():
    data = pd.read_excel('crashCourse/mlModels/activeStatsFilled.xlsx')
    rookieData = pd.read_excel('crashCourse/mlModels/rookiesFilled.xlsx')

    training = data.drop(['score', 'Name', 'wAv', 'draftYear', 'Yas', 'ProGames', 'yrs', 'totwAv'], axis=1)
    target = data['wAv']

    rookie_names = rookieData['Name'].values  # Extract the 'Name' column from rookieData
    rookies = rookieData.drop(['score', 'Name', 'wAv', 'draftYear', 'Yas', 'ProGames', 'yrs', 'totwAv'], axis=1)

    # Convert all column names to strings
    training.columns = training.columns.astype(str)
    rookies.columns = rookies.columns.astype(str)

    # Standardization 
    scaler = StandardScaler()
    training = scaler.fit_transform(training)  # Compute mean and var for training data and transform it
    rookies = scaler.transform(rookies)        # Use same mean and var to transform rookies data

    return training, target, rookies, rookie_names

# Define and compile the model
def build_model(input_shape):
    model = keras.Sequential([
        Dense(32, input_dim=input_shape, activation='relu'),
        Dropout(0.5),
        #Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        #Dropout(0.5),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.5),
        Dense(1, activation='relu')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model


def plot_history(history):
    # Summarize history for loss
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()


# Main routine
def main():
    # Load data
    training, target, rookies, rookie_names = load_data()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(training, target, test_size=0.1, random_state=42)

    # Build the model
    model = build_model(X_train.shape[1])

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=82, verbose=1, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, epochs=800, batch_size=32, verbose=1, 
        validation_data=(X_test, y_test), callbacks=[early_stop]
    )
    plot_history(history)

    # Evaluate the model performance
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nMean Squared Error: {mse:.2f}")

    # Make predictions
    predictions = model.predict(rookies)
    for name, pred in zip(rookie_names, predictions):
        print(f"{name}: {pred[0]:.2f}")

# Entry point
if __name__ == "__main__":
    main()
