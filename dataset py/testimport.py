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
                   'num_executed_instructions', 'execution_time', 'energy_efficiency',
                   'task_type', 'task_priority', 'task_status'])
df1 = df1.drop(columns=['vm_id', 'timestamp'])

#print(df1.isna().sum())
#print(df1.info())
#print(df1.describe())

ax=plt.subplots(1,1,figsize=(10,8))
sns.countplot(x='Anomaly status',data=df1)
plt.title("Anomaly status Count")
plt.show()