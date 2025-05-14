import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the dataset
parkinsons_data = pd.read_csv('parkinsons.csv')

# Preprocessing
X = parkinsons_data.drop(columns=['name', 'status'], axis=1)
y = parkinsons_data['status']

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train the SVM model
model = SVC(kernel='linear', probability=True, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Metrics
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_accuracy:.2f}")
print(f"Test Accuracy: {test_accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print("Classification Report:\n", classification_report(y_test, y_test_pred))

# Visualization: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_test_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Parkinsons', 'Parkinsons'], yticklabels=['No Parkinsons', 'Parkinsons'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Visualization: Bar Chart for Metrics
metrics = {'Accuracy': test_accuracy, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
names = list(metrics.keys())
values = list(metrics.values())

plt.bar(names, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
plt.title('Model Performance Metrics')
plt.ylim(0, 1)
plt.ylabel('Score')
plt.show()

# Save the model
filename = 'parkinsons_model_improved.sav'
pickle.dump({'model': model, 'scaler': scaler}, open(filename, 'wb'))

# Load and test with custom input
loaded_data = pickle.load(open(filename, 'rb'))
loaded_model = loaded_data['model']
loaded_scaler = loaded_data['scaler']

def predict_parkinsons(input_data):
    # Convert to numpy array and reshape
    input_data_np = np.asarray(input_data).reshape(1, -1)
    # Scale the input
    scaled_input = loaded_scaler.transform(input_data_np)
    # Predict
    prediction = loaded_model.predict(scaled_input)
    return "The Person has Parkinson's Disease" if prediction[0] == 1 else "The Person does not have Parkinson's Disease"

# Example custom input
custom_input = (197.07600, 206.89600, 192.05500, 0.00289, 0.00001, 0.00166, 0.00168, 0.00498,
                0.01098, 0.09700, 0.00563, 0.00680, 0.00802, 0.01689, 0.00339, 26.77500,
                0.422229, 0.741367, -7.348300, 0.177551, 1.743867, 0.085569)

result = predict_parkinsons(custom_input)
print("Prediction for custom input:", result)


