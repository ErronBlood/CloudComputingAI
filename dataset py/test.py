### Importar librerias
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

print(df.describe())

print(df.info())

#print(df.duplicated())

cat_col = [col for col in df.columns if df[col].dtype == 'object']
num_col = [col for col in df.columns if df[col].dtype != 'object']

print('Categorical columns:', cat_col)
print('Numerical columns:', num_col)

print(df[cat_col].nunique())

#Histograma del anomalias
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Anomaly status',data=df)
plt.title("Anomaly Status")
plt.show()
print(df['Anomaly status'].value_counts())

#print(round((df.isnull().sum() / df.shape[0]) * 100, 2))

df1 = df.drop(columns=['vm_id', 'timestamp'])
df1 = df1.dropna(subset=['cpu_usage', 'memory_usage', 'network_traffic', 'power_consumption',
                   'num_executed_instructions', 'execution_time', 'energy_efficiency',
                   'task_type', 'task_priority', 'task_status'])

print(df1.info())

print(df1.describe())

df.hist(['cpu_usage', 'memory_usage', 'network_traffic',
          'power_consumption', 'num_executed_instructions',
        'execution_time', 'energy_efficiency'], edgecolor='black', linewidth=1.0)
fig=plt.gcf()
fig.set_size_inches(10, 10)
#plt.show()

#Histograma del anomalias
ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Anomaly status',data=df)
plt.title("Anomaly Status")
#plt.show()
print(df1['Anomaly status'].value_counts())

#print(round((df1.isnull().sum() / df1.shape[0]) * 100, 2))



### Análisis de Correlación
# Exclusión de columnas no numéricas
df_numeric = df.select_dtypes(include='number')
correlation_matrix = df_numeric.corr()

# Configura el estilo de la figura
plt.figure(figsize=(10, 8))

# Crea un mapa de calor con seaborn
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)

# Añade título
plt.title("Matriz de Correlación")

# Muestra la visualización
plt.show()







