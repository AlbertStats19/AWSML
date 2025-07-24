#import pandas as pd
#import joblib
#import numpy as np
#import json
#import os
#import argparse
#from sklearn.datasets import load_iris # Para feature_names
#
#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR']) # Donde está el modelo y el scaler
#    parser.add_argument('--input-data-dir', type=str, default=os.environ.get('SM_CHANNEL_INPUT_DATA')) # Para los datos procesados
#    parser.add_argument('--config-dir', type=str, default=os.environ.get('SM_CHANNEL_CONFIG')) # Para prod_config.json
#    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
#    args = parser.parse_args()
#
#    # Cargar el modelo y el scaler
#    # Nota: El scaler no es estrictamente necesario para la predicción si ya los datos están escalados.
#    # Pero lo mantendremos para ser consistente con el flujo de artefactos.
#    model_path = os.path.join(args.model_dir, "model.joblib")
#    scaler_path = os.path.join(args.model_dir, "scaler.joblib")
#    model = joblib.load(model_path)
#    scaler = joblib.load(scaler_path) # Cargar el scaler, aunque no lo usemos para la predicción en sí (los datos ya están escalados)
#
#    # Cargar la configuración de batch prediction
#    config_path = os.path.join(args.config_dir, 'prod_config.json')
#    with open(config_path, 'r') as f:
#        config = json.load(f)
#    batch_config = config["BATCH_PREDICTION"]
#    sample_rate = batch_config["SAMPLE_RATE"]
#    output_file_name = batch_config["OUTPUT_FILE_NAME"]
#
#    # Cargar los datos procesados completos para muestreo
#    processed_data_path = os.path.join(args.input_data_dir, 'iris_processed.csv')
#    df_processed = pd.read_csv(processed_data_path)
#    X_processed = df_processed.drop('target', axis=1).values # Convertir a numpy array
#
#    # Realizar el muestreo
#    n_samples = int(len(X_processed) * sample_rate)
#    np.random.seed(42)
#    sample_indices = np.random.choice(len(X_processed), n_samples, replace=False)
#
#    X_batch = X_processed[sample_indices] # Ya está escalado
#
#    # Realizar predicciones
#    predicciones_batch = model.predict(X_batch)
#
#    # Crear DataFrame de salida
#    feature_names = load_iris().feature_names # Obtener nombres de características
#    X_df_batch = pd.DataFrame(X_batch, columns=feature_names)
#    X_df_batch["prediction"] = predicciones_batch
#
#    # Guardar predicciones batch
#    output_path = os.path.join(args.output_data_dir, output_file_name)
#    X_df_batch.to_csv(output_path, index=False, sep=";", decimal=",")
#    print(f"Batch predictions saved to {output_path}")

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
