import pandas as pd
import numpy as np

from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dropout
from tensorflow.keras.regularizers import l2

from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


import matplotlib.pyplot as plt


# Load and preprocess data
def load_data():
    data = pd.read_excel('data/activeStatsFilled.xlsx')
    rookieData = pd.read_excel('data/rookiesFilled.xlsx')

    player_names = data['Name'].values  # This array aligns with your training data
    training = data.drop(['score', 'Name', 'wAv', 'draftYear', 'yrs', 'totwAv', 'Pick'], axis=1)
    target = data['wAv']

    rookie_names = rookieData['Name'].values  # Extract the 'Name' column from rookieData
    rookies = rookieData.drop(['score', 'Name', 'wAv', 'draftYear', 'yrs', 'totwAv', 'Pick'], axis=1)

    # Convert all column names to strings (For the scaler)
    training.columns = training.columns.astype(str)
    rookies.columns = rookies.columns.astype(str)

    # Standardization 
    scaler = StandardScaler()
    training = scaler.fit_transform(training)  # Compute mean and var for training data and transform it
    rookies = scaler.transform(rookies)        # Use same mean and var to transform rookies data

    # plot_pca(training, player_names)

    return training, target, rookies, rookie_names, player_names




# Plotting the PCA
def plot_pca(training_data, player_names):
    # Perform PCA
    pca = PCA(n_components=2)  # Only the first two principal components are retained
    principalComponents = pca.fit_transform(training_data)

    # Creating a DataFrame for the principal components to facilitate plotting
    principalDf = pd.DataFrame(data=principalComponents, columns=['PC1', 'PC2'])
    principalDf['Name'] = player_names

    # Plot the PCA
    plt.figure(figsize=(12, 8))
    for i in range(len(principalDf)):
        plt.scatter(principalDf.at[i, 'PC1'], principalDf.at[i, 'PC2'], c='b', marker='o', alpha=0.5)
        plt.text(principalDf.at[i, 'PC1'], principalDf.at[i, 'PC2'], principalDf.at[i, 'Name'], fontsize=9, ha='right')  # Adjusting text properties for clarity

    plt.title('Principal Component Analysis (PCA) of Training Data')
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.grid()
    plt.show()




# Define and compile the model
def build_model(input_shape):
    model = keras.Sequential([
        Dense(32, input_dim=input_shape, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(32, activation='relu', kernel_regularizer=l2(0.01)),
        Dropout(0.2),
        Dense(1, activation='relu')
    ])

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_squared_error'])
    return model





def plot_history(history):
    # Summarize history for loss
    plt.figure(figsize=(10, 5))

    plt.plot(history.history['loss'], label='Training Loss', color='teal')
    plt.plot(history.history['val_loss'], label='Validation Loss', color='deeppink')
    plt.title('Model Loss over epochs')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.grid(True)
    plt.show()




def random_forest_regression(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None):
    # Initialize the Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42)

    # Fit the model to the training data
    rf.fit(X_train, y_train)

    # Predict on the test data
    y_pred = rf.predict(X_test)

    # Calculate the mean squared error
    mse = mean_squared_error(y_test, y_pred)
    print(f"Random Forest Regressor MSE: {mse:.2f}")

    return rf

def fit_nearest_neighbors(current_players, n_neighbors=1):
    nn = NearestNeighbors(n_neighbors=n_neighbors, algorithm='ball_tree')
    nn.fit(current_players)
    return nn

def find_closest_players(nn_model, current_player_names, rookies):
    distances, indices = nn_model.kneighbors(rookies)
    closest_players = current_player_names[indices]
    return closest_players, distances



# Main routine
def main():
    # Load data
    training, target, rookies, rookie_names, player_names = load_data()
  
    """
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(training, target, test_size=0.2, random_state=10)
    
    #############################################################################################################
    ######################################## Neural network model ###############################################
    #############################################################################################################
    
    # Build the model
    model = build_model(X_train.shape[1])

    # Train the model
    early_stop = EarlyStopping(monitor='val_loss', patience=200, verbose=1, restore_best_weights=True)

    history = model.fit(
        X_train, y_train, epochs=800, batch_size=40, verbose=1, 
        validation_data=(X_test, y_test), callbacks=[early_stop]
    )

    #plot_history(history)

    # Evaluate the model performance
    loss, mse = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nMean Squared Error: {mse:.2f}")

    # Make predictions
    predictions = model.predict(rookies)
    for name, pred in zip(rookie_names, predictions):
        print(f"{name}: {pred[0]:.2f}")
        
    
    #############################################################################################################
    #################################### Random forrest regressor ###############################################
    #############################################################################################################
    rf_model = random_forest_regression(X_train, y_train, X_test, y_test, n_estimators=100, max_depth=None)

    
    # Make predictions on rookie data
    rookie_predictions = rf_model.predict(rookies)
    for name, pred in zip(rookie_names, rookie_predictions):
        print(f"{name}: {pred:.2f}")
    """
    nn_model = fit_nearest_neighbors(training)
    closest_players, distances = find_closest_players(nn_model, player_names, rookies)
    
    # Combine into a list of tuples
    combined_list = zip(rookie_names, closest_players.flatten(), distances.flatten())
    
    # Sort by distance (the third element in each tuple)
    sorted_list = sorted(combined_list, key=lambda x: x[2])
    
    # Now print in sorted order
    for rookie, closest_player, distance in sorted_list:
        print(f"{rookie} - {closest_player}, distance: {distance:.2f}")


if __name__ == "__main__":
    main()
