import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Ruta FIJA al input del dataset
input_file_path = "/opt/ml/processing/input/iris_raw.csv"
output_data_path = "/opt/ml/processing/output/iris_processed.csv"
scaler_path = "/opt/ml/processing/scaler/scaler.joblib"

# Cargar datos Iris crudos
df = pd.read_csv(input_file_path)

X = df.drop('target', axis=1)  # Características
y = df['target']  # Target

# Preprocesar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reconstruir el dataframe
processed_df = pd.DataFrame(X_scaled, columns=X.columns)
processed_df['target'] = y

# Guardar archivos
processed_df.to_csv(output_data_path, index=False)
joblib.dump(scaler, scaler_path)

print(f"✅ Procesado OK: {output_data_path}")
print(f"✅ Scaler OK: {scaler_path}")
