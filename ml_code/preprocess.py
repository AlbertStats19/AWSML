#import pandas as pd
#from sklearn.preprocessing import StandardScaler
#import joblib
#import os
#import argparse

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    #parser.add_argument('--input-data-dir', type=str, default=os.environ.get('SM_CHANNEL_INPUT_DATA'))
#    parser.add_argument('--input-data-dir', type=str, default=os.environ.get('SM_CHANNEL_INPUT'))
#    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
#    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR')) # Para guardar el scaler
#    args = parser.parse_args()
#
#    # Cargar datos Iris crudos
#    input_file_path = os.path.join(args.input_data_dir, 'iris_raw.csv')
#    df = pd.read_csv(input_file_path)
#
#    X = df.drop('target', axis=1) # Las características
#    y = df['target'] # El target
#
#    # Simula el preprocesamiento de los datos: escalar características
#    scaler = StandardScaler()
#    X_scaled = scaler.fit_transform(X)
#
#    # Recombinar X_scaled y y en un DataFrame para pasarlo al siguiente paso
#    processed_df = pd.DataFrame(X_scaled, columns=X.columns)
#    processed_df['target'] = y
#
#    # Guardar los datos procesados
#    processed_data_path = os.path.join(args.output_data_dir, 'iris_processed.csv')
#    processed_df.to_csv(processed_data_path, index=False)
#    print(f"Processed data saved to {processed_data_path}")
#
#    # Guardar el scaler entrenado (será necesario para la inferencia)
#    scaler_path = os.path.join(args.model_dir, "scaler.joblib")
#    joblib.dump(scaler, scaler_path)
#    print(f"Scaler saved to {scaler_path}")

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
