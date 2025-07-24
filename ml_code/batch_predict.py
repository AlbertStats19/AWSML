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
import pandas as pd
import joblib
import numpy as np
import json
from sklearn.datasets import load_iris

if __name__ == '__main__':
    print("==> Iniciando predicción por lotes...")

    model_dir = os.environ['SM_MODEL_DIR']
    input_data_dir = os.environ['SM_CHANNEL_INPUT_DATA']
    config_dir = os.environ['SM_CHANNEL_CONFIG']
    output_data_dir = os.environ['SM_OUTPUT_DATA_DIR']

    # Cargar modelo y scaler
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    scaler = joblib.load(os.path.join(model_dir, "scaler.joblib"))

    # Leer configuración
    config_path = os.path.join(config_dir, 'prod_config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    batch_config = config["BATCH_PREDICTION"]
    sample_rate = batch_config["SAMPLE_RATE"]
    output_file_name = batch_config["OUTPUT_FILE_NAME"]

    # Leer los datos procesados
    df = pd.read_csv(os.path.join(input_data_dir, 'iris_processed.csv'))
    X = df.drop('target', axis=1).values

    # Muestreo
    np.random.seed(42)
    n_samples = int(len(X) * sample_rate)
    sample_idx = np.random.choice(len(X), n_samples, replace=False)
    X_batch = X[sample_idx]

    # Predicciones
    preds = model.predict(X_batch)

    # Resultado
    feature_names = load_iris().feature_names
    df_out = pd.DataFrame(X_batch, columns=feature_names)
    df_out["prediction"] = preds

    # Guardar
    output_path = os.path.join(output_data_dir, output_file_name)
    df_out.to_csv(output_path, index=False, sep=";", decimal=",")

    print(f"✅ Predicciones guardadas en {output_path}")