#Loading all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

#loading the dataset
data = pd.read_csv('water_potability.csv')
print(data.head())

#checking for missing values
print("Missing values before imputation:\n", data.isnull().sum())

imputer = SimpleImputer(strategy='median')
data.iloc[:,:-1] = imputer.fit_transform(data.iloc[:, :-1])
print("Missing Values after imputation:\n", data.isnull().sum())
print(f'Dataset shape before dropping duplicates: {data.shape}')

#Dropping duplicates

data = data.drop_duplicates()
print(f'Dataset shape after dropping duplicates: {data.shape}')

#Feature Selection
X = data.drop(columns=['Potability']) #features
y = data['Potability'] #target value

#Spliting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Model training
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)

#Model Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")


#finding the best k value
best_k = 1
best_accuracy = 0

for k in range(1, 21):  # Test k values from 1 to 20
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"k = {k}, Accuracy = {accuracy:.2f}")
    
    if accuracy > best_accuracy:
        best_k = k
        best_accuracy = accuracy

print(f"Best k: {best_k} with Accuracy: {best_accuracy:.2f}")

#Normalizing or Standardizing the features

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Model training after normalization
k = 28
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)

#Model Evaluation after normalization
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')

print(f"Standardized Accuracy: {accuracy:.2f}")
print(f"Standardized Precision: {precision:.2f}")
print(f"Standardized Recall: {recall:.2f}")

#Confusion Matrix (Visulaization 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Potable', 'Potable'], yticklabels=['Not Potable', 'Potable'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Not Potable', 'Potable']))

#Visulaization 2
# Plot feature distributions grouped by Potability
features = ['ph', 'Hardness', 'Solids', 'Chloramines', 'Sulfate', 
            'Conductivity', 'Organic_carbon', 'Trihalomethanes', 'Turbidity']

plt.figure(figsize=(20, 15))
for i, feature in enumerate(features, 1):
    plt.subplot(3, 3, i)
    sns.histplot(data=data, x=feature, hue='Potability', kde=True, palette='viridis')
    plt.title(f"Distribution of {feature} by Potability")
    plt.xlabel(feature)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.show()

#saving the model using pickle
import pickle

with open('knn_water_potability_model.pkl', 'wb') as file:
    pickle.dump(knn, file)

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)

print("Model and scaler saved successfully!")