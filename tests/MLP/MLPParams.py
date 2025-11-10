#Modelos

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import tree

#Metricas

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
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

mlp = MLPClassifier()

param_grid = {
    'hidden_layer_sizes': [(50,50,50),
                           (100,50),
                           (150)],
    'activation': ['relu','tanh'],  # Regularization strength
    'solver': ['lbfgs', 'sgd', 'adam'],  # Optimization algorithm  
    'alpha': [0.001],
    'learning_rate' : ['constant', 'adaptive'],
    'learning_rate_init' : [0.01, 0.1, 1],
    'max_iter' : [800, 900, 1000],
    'early_stopping' : [False]

}

comb = RandomizedSearchCV(mlp, param_grid, cv = 5, scoring = 'recall', n_jobs=-1, n_iter=20, random_state=42, verbose=2)

comb.fit(X_train_scaled,y_train)

print("Mejor combinacion de parametros: {:}".format(comb.best_params_))
print("Puntaje de mejores parametros: {:.2}".format(comb.best_score_))

bestComb = comb.best_estimator_

y_pred = bestComb.predict(X_test_scaled)

rf_score = recall_score(y_true=y_test, y_pred= y_pred)

print('Accuracy de LogisticRegression sobre el conjunto de prueba es: {:.2f}'.format(rf_score)) 
cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)

print(classification_report(y_test, y_pred))
print(X_resampled.columns)