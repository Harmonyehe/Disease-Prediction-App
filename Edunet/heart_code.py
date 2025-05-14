import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

def load_and_preprocess_data(file_path):
    # Loading the dataset
    data = pd.read_csv(file_path)
    
    # Separate features and labels
    X = data.drop(columns=['target'])
    y = data['target']
    
    # Handle missing values by replacing 0 with NaN and filling with column mean
    X.replace(0, np.nan, inplace=True)
    X.fillna(X.mean(), inplace=True)
    
    return X, y

def scale_data(X_train, X_test):
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_test_scaled, scaler

def train_gradient_boosting(X_train, y_train):
    # Initialize and train the model
    model = GradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(y_test, y_pred):
    # Performance Evaluation
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Classification Report:\n", classification_report(y_test, y_pred))

    # Confusion Matrix Visualization
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['No Heart Disease', 'Heart Disease'], yticklabels=['No Heart Disease', 'Heart Disease'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()

    # Plotting model performance metrics
    metrics = {'Accuracy': accuracy, 'F1 Score': f1, 'Precision': precision, 'Recall': recall}
    names = list(metrics.keys())
    values = list(metrics.values())

    plt.bar(names, values, color=['skyblue', 'lightgreen', 'orange', 'pink'])
    plt.title('Model Performance Metrics')
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.show()

def save_model(model, scaler, filename='heart_disease_model.sav'):
    # Save the trained model and scaler
    joblib.dump((model, scaler), filename)
    print(f"Model saved to {filename}")

def load_model(filename='heart_disease_model.sav'):
    # Load the saved model and scaler
    model, scaler = joblib.load(filename)
    return model, scaler

def predict_heart_disease(model, scaler, custom_input, X_columns):
    # Prepare the input data for prediction
    input_df = pd.DataFrame([custom_input], columns=X_columns)
    input_df.fillna(X.mean(), inplace=True)
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    return "Heart Disease" if prediction == 1 else "No Heart Disease"

def main():
    # Step 1: Load and preprocess the data
    data_path = os.path.join(os.path.dirname(__file__), 'heart.csv')
    X, y = load_and_preprocess_data(data_path)
    
    # Step 2: Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Step 3: Scale the data
    X_train_scaled, X_test_scaled, scaler = scale_data(X_train, X_test)
    
    # Step 4: Train the Gradient Boosting model
    model = train_gradient_boosting(X_train_scaled, y_train)
    
    # Step 5: Make predictions and evaluate the model
    y_pred = model.predict(X_test_scaled)
    evaluate_model(y_test, y_pred)
    
    # Step 6: Save the model and scaler for future use
    save_model(model, scaler)
    
    # Step 7: Load the model and make predictions for custom input
    model, scaler = load_model()
    
    custom_input = {
        'age': 45,
        'sex': 1,
        'cp': 3,
        'trestbps': 120,
        'chol': 233,
        'fbs': 0,
        'restecg': 1,
        'thalach': 150,
        'exang': 0,
        'oldpeak': 0.2,
        'slope': 2,
        'ca': 0,
        'thal': 2
    }
    
    result = predict_heart_disease(model, scaler, custom_input, X.columns)
    print("Prediction for custom input:", result)

# Run the program
main()

