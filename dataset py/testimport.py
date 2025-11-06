import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import kagglehub
from kagglehub import KaggleDatasetAdapter

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from sklearn import tree
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

# Visualize
ax = plt.subplots(1, 1, figsize=(10, 8))
sns.countplot(x='Anomaly status', data=dfResampled)
plt.title("Anomaly status Count")
plt.show()

print(f"Original shape: {df1.shape}")
print(f"Resampled shape: {dfResampled.shape}")

x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.20)

clf = tree.DecisionTreeClassifier(criterion='entropy', splitter='best', max_depth=20)
clf = clf.fit(x_train,y_train)

#Evaluación en el conjunto de prueba
y_pred = clf.predict(x_test)

print('Accuracy de DecisionTreeClassifier sobre el conjunto de prueba es: {:.2f}'.format(clf.score(x_test, y_test))) 
cmatrix = confusion_matrix(y_test, y_pred)
print(cmatrix)

print(classification_report(y_test, y_pred))
print(X_resampled.columns)


