import pandas as pd
from sklearn.model_selection import train_test_split
import os

# Rutas estándar en SageMaker
input_file_path = "/opt/ml/processing/input/iris_processed.csv"
output_dir = "/opt/ml/processing/output"

# Cargar datos procesados
df = pd.read_csv(input_file_path)

X = df.drop('target', axis=1)
y = df['target']

# Dividir la data en entrenamiento y prueba (50/50 como en tu ejemplo)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42, stratify=y)

# Guardar los conjuntos de datos divididos
pd.DataFrame(X_train, columns=X.columns).to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
pd.DataFrame(y_train, columns=['target']).to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
pd.DataFrame(y_test, columns=['target']).to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)

print("✅ Data split and saved correctamente en:")
print(f"- {os.path.join(output_dir, 'X_train.csv')}")
print(f"- {os.path.join(output_dir, 'X_test.csv')}")
print(f"- {os.path.join(output_dir, 'y_train.csv')}")
print(f"- {os.path.join(output_dir, 'y_test.csv')}")
