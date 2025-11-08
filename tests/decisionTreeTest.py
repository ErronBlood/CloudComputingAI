import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, recall_score
from sklearn import tree

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

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

#Mejores parametros encontrados
dt = tree.DecisionTreeClassifier(criterion='entropy', max_depth=23, min_samples_leaf=75, min_samples_split=10, splitter='random')

dt.fit(X_train_scaled, y_train)

y_pred = dt.predict(X_test_scaled)

cmatrix = confusion_matrix(y_test, y_pred)
labels = np.unique(y_test)

score = recall_score(y_true= y_test, y_pred=y_pred)

df_cm = pd.DataFrame(cmatrix, index=labels, columns=labels)
plt.figure(figsize=(8, 6))
sns.heatmap(df_cm, annot=True, cmap='Blues', fmt='d')
plt.title("Matriz de Confusión")
plt.xlabel("Etiqueta Predicha")
plt.ylabel("Etiqueta Verdadera")
plt.show()

print('Recall de DecisionTree sobre el conjunto de prueba es: {:.2f}'.format(score)) 

print(cmatrix)

print(classification_report(y_test, y_pred))
print(X_resampled.columns)