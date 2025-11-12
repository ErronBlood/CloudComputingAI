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

dfResampled = pd.DataFrame(X, columns=X.columns)
dfResampled['Anomaly status'] = y

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

# Apply SMOTE
smote = SMOTE(random_state=42)
x_trainR, y_trainR = smote.fit_resample(x_train, y_train)

# Recreate the balanced dataframe


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(x_trainR)
X_test_scaled = scaler.transform(x_test)

#Seleccion e instanciacion del modelo
# Penalty = l2
# C = 1.0
# solver = lbfgs
# Max_iter = 100
lr = LogisticRegression()

#Entrenamiento del modelo con banco de entrenamieno escalado
lr.fit(X_train_scaled, y_trainR)

#Realizacion de predicciones con datos de entrenamiento y prueba
y_pred_train = lr.predict(X_train_scaled)
y_pred_test = lr.predict(X_test_scaled)

#Calculo de porcentaje de recall para ambas predicciones
lr_train_f1= f1_score(y_true= y_trainR, y_pred = y_pred_train) * 100
lr_test_f1 = f1_score(y_true= y_test, y_pred = y_pred_test) * 100

print('Sensibilidad a datos de entrenamiento: {:.2f}'.format(lr_train_f1))
print('Sensibilidad a datos de prueba: {:.2f}\n'.format(lr_test_f1))
print(classification_report(y_test, y_pred_test))

cmatrix = confusion_matrix(y_test, y_pred_test)
labels = np.unique(y_test)
df_cm = pd.DataFrame(cmatrix, index=labels, columns=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de Confusión para Datos de Prueba")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()