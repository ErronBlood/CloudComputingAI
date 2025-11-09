#Modelos

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

#Metricas

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, recall_score

#Ajuste de datos

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

#Utilidades

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "sandhyapeesara/cloud-anomaly-data",
  "Cloud_Anomaly_Dataset.csv"
)

df1 = df.dropna(subset=['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
                   'num_executed_instructions', 'execution_time', 'energy_efficiency'])
df1 = df1.drop(columns=['vm_id', 'timestamp', 'task_type', 'task_priority', 'task_status'])

# Split features (X) and target (y)
X = df1.drop(columns=['Anomaly status'])
y = df1['Anomaly status']

# Apply SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Recreate the balanced dataframe
dfResampled = pd.DataFrame(X_resampled, columns=X.columns)
dfResampled['Anomaly status'] = y_resampled

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

dt = tree.DecisionTreeClassifier()

# Definicmos los valores posibles para los hiperparametros

param_grid = {
    'criterion': ['entropy'],  # Regularization strength
    'splitter': ['best', 'random'],
    'min_samples_split' : [2,4,8,10],  # Regularization type
    'max_depth': [20,23,26,29],  # Optimization algorithm  
    'min_samples_leaf': [50,75]  # Maximum iterations
}

# Se realiza la busqueda de parametros con GridSearchCV, juzgando por la estadistica de sensibilidad
comb = GridSearchCV(dt, param_grid, cv = 5, scoring = 'recall', n_jobs=-1)

comb.fit(X_train_scaled,y_train)

print("Mejor combinacion de parametros: {:.2}".format(comb.best_params_))
print("Puntaje de mejores parametros: {:.2}".format(comb.best_score_))

bestComb = comb.best_estimator_

y_pred = bestComb.predict(X_test_scaled)

dt_score = recall_score(y_true=y_test, y_pred= y_pred)

print('Accuracy de LogisticRegression sobre el conjunto de prueba es: {:.2f}'.format(dt_score)) 
cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)

print(classification_report(y_test, y_pred))
print(X_resampled.columns)