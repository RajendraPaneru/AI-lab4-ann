import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# 1. Create a sample binary classification dataset
X, y = make_classification(n_samples=500, n_features=5, n_informative=3, n_redundant=0, random_state=42)

# 2. Preprocess the data
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Build the ANN model
model = Sequential()
model.add(Dense(16, input_dim=X.shape[1], activation='relu'))  # input layer + 1st hidden layer
model.add(Dense(8, activation='relu'))                         # 2nd hidden layer
model.add(Dense(1, activation='sigmoid'))                      # output layer (binary)

# 4. Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# 5. Train the model
model.fit(X_train, y_train, epochs=20, batch_size=10, validation_split=0.1)

# 6. Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# 7. Predict new input
new_input = np.array([[0.1, -1.2, 1.4, 0.2, 0.9]])  # Example input (scaled)
new_input = scaler.transform(new_input)
prediction = model.predict(new_input)
print("Predicted class:", int(prediction[0][0] > 0.5))
