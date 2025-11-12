#Modelos

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

#Metricas

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, f1_score

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


smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Recreate the balanced dataframe
dfResampled = pd.DataFrame(X_resampled, columns=X.columns)
dfResampled['Anomaly status'] = y_resampled

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_train)
X_test_scaled = scaler.transform(x_test)

# Definimos los valores posibles para realizar las combinaciones de hiperparametros
param_grid = {
    'C': [0.00001, 0.0001, 0.001],  
    'penalty': ['l1', 'l2'],  
    'solver': ['liblinear', 'saga'],  
    'max_iter': [500,600,700,800]
}

lr = LogisticRegression()

# Se realiza la busqueda de parametros con GridSearchCV, juzgando por la estadistica de sensibilidad
comb = GridSearchCV(lr, param_grid, cv = 5, scoring = 'f1', n_jobs=-1)

comb.fit(X_train_scaled,y_train)

print("Mejor combinacion de parametros: {:.2} ", format(comb.best_params_))
print("Puntaje de mejores parametros: {:.2}".format(comb.best_score_))

bestComb = comb.best_estimator_

y_pred = bestComb.predict(X_test_scaled)

dt_score = f1_score(y_true=y_test, y_pred= y_pred)

print('Accuracy de LogisticRegression sobre el conjunto de prueba es: {:.2f}'.format(dt_score)) 
cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)

print(classification_report(y_test, y_pred))