import os
import tarfile
import pandas as pd
import joblib
import json

print("==> Iniciando predicción por lotes...")

# Rutas
compressed_model_path = "/opt/ml/processing/model/model.tar.gz"
extracted_model_path = "/opt/ml/processing/model/model.joblib"
input_data_path = "/opt/ml/processing/input/iris_raw.csv"
config_path = "/opt/ml/processing/config/prod_config.json"
output_path = "/opt/ml/processing/output/batch_pred.csv"

# Descomprimir el modelo
print("==> Descomprimiendo modelo...")
with tarfile.open(compressed_model_path, "r:gz") as tar:
    tar.extractall(path="/opt/ml/processing/model")

# Cargar modelo descomprimido
print("==> Cargando modelo descomprimido...")
model = joblib.load(extracted_model_path)

# Cargar datos
df = pd.read_csv(input_data_path)
df = df.get([x for x in df.columns if x not in ["target"]]).copy()

# Hacer predicciones
df["prediction"] = model.predict(df)

# Leer configuración
with open(config_path, "r") as f:
    config = json.load(f)

sample_rate = config["BATCH_PREDICTION"]["SAMPLE_RATE"]
output_file_name = config["BATCH_PREDICTION"]["OUTPUT_FILE_NAME"]

# Tomar muestra
df_sample = df.sample(frac=sample_rate)

# Guardar predicciones
df_sample.to_csv(output_path, index=False)
print(f"==> Predicciones guardadas en: {output_path}")
