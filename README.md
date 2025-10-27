# 🌸 Iris Flower Classification

## 🎯 Objective
Classify iris flowers into three species (**Setosa**, **Versicolor**, **Virginica**) based on petal and sepal measurements.

## 🧩 Dataset
The Iris dataset from scikit-learn (`sklearn.datasets.load_iris`).

## 🧠 Steps
1. Load dataset
2. Visualize features
3. Split into training/test sets
4. Scale features
5. Train KNN classifier
6. Evaluate and save results

## 📊 Output Files
- `outputs/figures/iris_pairplot.png`
- `outputs/results/evaluation.txt`
- `outputs/models/iris_knn_model.pkl`

## ⚙️ How to Run
```bash
pip install -r requirements.txt
cd src
python iris_classification.py
