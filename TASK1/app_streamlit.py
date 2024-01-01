import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Iris Flower Classification App", page_icon="🌺", layout="wide")

# Load the Iris dataset
iris = load_iris()
data = pd.DataFrame(data=np.c_[iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the K-Nearest Neighbors (KNN) classifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train_scaled, y_train)

# Pairplot visualization
st.title("Iris Pairplot Visualization")
st.pyplot(sns.pairplot(data, hue='target', palette={0: 'red', 1: 'green', 2: 'blue'}, hue_order=[0, 1, 2]))

# Input form
st.title("Iris Flower Classification")
sepal_length = st.number_input("Enter sepal length:")
sepal_width = st.number_input("Enter sepal width:")
petal_length = st.number_input("Enter petal length:")
petal_width = st.number_input("Enter petal width:")

# Standardize the input features using the same scaler
input_data = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])

# Make a prediction
prediction = knn.predict(input_data)[0]

# Map the prediction to the corresponding species name
species_names = iris.target_names
predicted_species = species_names[prediction]

# Display result
st.write(f"The predicted species is: {predicted_species}")

# Model Evaluation
y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

st.title("Model Evaluation")
st.write(f"Accuracy: {accuracy}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
st.write("Classification Report:")
st.write(class_report)

