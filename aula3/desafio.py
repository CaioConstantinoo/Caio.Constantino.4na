import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Carregar o conjunto de dados Wine
wine = load_wine()
X = wine.data
y = wine.target

# Dividir os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Criar o pipeline com pré-processamento e o classificador KNN
pipeline = Pipeline([
    ('scaler', StandardScaler()),  # Padronização dos dados
    ('knn', KNeighborsClassifier(n_neighbors=5))  # Classificador KNN
])

# Treinar o modelo
pipeline.fit(X_train, y_train)

# Fazer predições no conjunto de teste
y_pred = pipeline.predict(X_test)

# Avaliar o modelo usando diversas métricas
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

print("Acurácia:")
print(accuracy_score(y_test, y_pred))
