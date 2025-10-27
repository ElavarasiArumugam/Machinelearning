# src/iris_classification.py

import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib

# === 1. Folder setup ===
output_dir = '../outputs'
figures_dir = os.path.join(output_dir, 'figures')
results_dir = os.path.join(output_dir, 'results')
models_dir = os.path.join(output_dir, 'models')

for folder in [figures_dir, results_dir, models_dir]:
    os.makedirs(folder, exist_ok=True)

print("Output folders ready!")

# === 2. Load dataset ===
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

df = pd.DataFrame(X, columns=feature_names)
df['species'] = y

# === 3. Visualization with unique filename ===
timestamp = int(time.time())
plot_filename = f'iris_pairplot_{timestamp}.png'
sns.pairplot(df, hue='species')
plt.savefig(os.path.join(figures_dir, plot_filename))
plt.close()
print(f"Pairplot saved as: {plot_filename}")

# === 4. Train/test split (fixed by default) ===
use_random_split = False  # Change to True to randomize split each run
random_state = None if use_random_split else 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

# === 5. Scale Data ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Train KNN Model ===
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# === 7. Cross Validation ===
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
cv_accuracy = cv_scores.mean()

# === 8. Evaluate ===
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=target_names)

# === 9. Save Results ===
result_filename = f'evaluation_{timestamp}.txt'
with open(os.path.join(results_dir, result_filename), 'w') as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"Cross-Validation Accuracy: {cv_accuracy:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"Evaluation saved as: {result_filename}")

# === 10. Save Model ===
model_filename = f'iris_knn_model_{timestamp}.pkl'
joblib.dump(model, os.path.join(models_dir, model_filename))
print(f"Model saved as: {model_filename}")

# === 11. Interactive Prediction ===
print("\nPredict a New Iris Flower")
try:
    sepal_length = float(input("Enter sepal length (cm): "))
    sepal_width = float(input("Enter sepal width (cm): "))
    petal_length = float(input("Enter petal length (cm): "))
    petal_width = float(input("Enter petal width (cm): "))

    new_sample = scaler.transform([[sepal_length, sepal_width, petal_length, petal_width]])
    predicted_species = target_names[model.predict(new_sample)[0]]

    print(f"\nThe predicted species is: {predicted_species}")

except ValueError:
    print("Invalid input! Please enter numeric values for all measurements.")
