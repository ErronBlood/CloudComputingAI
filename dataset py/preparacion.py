import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import kagglehub

from kagglehub import KaggleDatasetAdapter

#Cagando datos
df = kagglehub.dataset_load(
  KaggleDatasetAdapter.PANDAS,
  "sandhyapeesara/cloud-anomaly-data",
  "Cloud_Anomaly_Dataset.csv"
)

#Informacion de los datos
#print(df.info())

#Resumen estadistico de las variables numericas de los datos
#print(df.describe())

#Histograma del estado de las anomalias
#ax=plt.subplots(1,1,figsize=(10,8))
#sns.countplot(x='Anomaly status',data=df)
#plt.title("Anomaly status")
#plt.show()#

#Histograma de frencuencias de las variables numericas
#df_var = df.drop(['Anomaly status'], axis=1)
#df_var.hist(edgecolor='black', linewidth=1.2)
#fig=plt.gcf()
#fig.set_size_inches(12, 10)
#plt.show()

df1 = df.dropna(subset=['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
                   'num_executed_instructions', 'execution_time', 'energy_efficiency',
                   'task_type', 'task_priority', 'task_status'])
df1 = df1.drop(columns=['vm_id', 'timestamp'])

#print(df1.isna().sum())

#print(df1.describe())

#Histograma del estado de las anomalias
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Anomaly status',data=df1)
plt.title("Anomalies")
#plt.show()#

#Histograma de frencuencias de las variables numericas
df1_var = df1.drop(['Anomaly status'], axis=1)
df1_var.hist(edgecolor='black', linewidth=1.2)
fig=plt.gcf()
fig.set_size_inches(12, 10)
#plt.show()

#cols = [
#    "cpu_usage", "memory_usage", "network_traffic", 
#    "power_consumption", "num_executed_instructions",   
#    "execution_time", "energy_efficiency","Anomaly status"
#]
#
#scaler = MinMaxScaler()
#df_scaled = df1.copy()
#df_scaled[cols] = scaler.fit_transform(df1[cols])
#
#print(df_scaled[cols].describe())
#
##Histograma de frencuencias de las variables numericas
#df_var = df_scaled[cols].drop(['Anomaly status'], axis=1)
#df_var.hist(edgecolor='black', linewidth=1.2)
#fig=plt.gcf()
#fig.set_size_inches(12, 10)
#plt.show()

df1.plot(kind='box', subplots=True, layout=(3,3), figsize=(12,8))
plt.suptitle("Distribución por variable (Boxplots normalizados)")
#plt.show()

### Análisis de Correlación
# Exclusión de columnas no numéricas
df1_numeric = df1.select_dtypes(include='number')
correlation_matrix = df1_numeric.corr()

# Configura el estilo de la figura
plt.figure(figsize=(10, 8))

# Crea un mapa de calor con seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Añade título
plt.title("Matriz de Correlación")

# Muestra la visualización
#plt.show()

X_train = df1.drop(['Anomaly status','task_type', 'task_priority', 'task_status'], axis=1) # Separa las variables predictoras de las variable a predecir
y_train = df1['Anomaly status']

f, ax = plt.subplots(figsize=(10, 8))
corr = X_train.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=bool), 
          cmap=sns.diverging_palette(220, 10, as_cmap=True),square=True, ax=ax, linewidths=.5)
#plt.show()


# Crear copia del dataframe
df_encoded = df1.copy()

# Crear codificador
le = LabelEncoder()

# Convertir columnas categóricas a números
for col in ['task_type', 'task_priority', 'task_status']:
    df_encoded[col] = le.fit_transform(df_encoded[col])

# Verificar conversión
print(df_encoded[['task_type', 'task_priority', 'task_status']].describe())
print(df_encoded[['task_type', 'task_priority', 'task_status']].info())

print(df_encoded.info())