#import pandas as pd
#from sklearn.metrics import accuracy_score, f1_score
#import joblib
#import os
#import argparse
#import json

#if __name__ == '__main__':
#    parser = argparse.ArgumentParser()
#    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
#    parser.add_argument('--input-data-dir', type=str, default=os.environ['SM_CHANNEL_INPUT_DATA']) # Para datos de test
#    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
#    args = parser.parse_args()
#
#    # Cargar el modelo
#    model_path = os.path.join(args.model_dir, "model.joblib")
#    model = joblib.load(model_path)
#
#    # Cargar datos de entrenamiento y prueba
#    X_train_path = os.path.join(args.input_data_dir, 'X_train.csv')
#    y_train_path = os.path.join(args.input_data_dir, 'y_train.csv')
#    X_test_path = os.path.join(args.input_data_dir, 'X_test.csv')
#    y_test_path = os.path.join(args.input_data_dir, 'y_test.csv')
#
#    X_train = pd.read_csv(X_train_path)
#    y_train = pd.read_csv(y_train_path).squeeze("columns")
#    X_test = pd.read_csv(X_test_path)
#    y_test = pd.read_csv(y_test_path).squeeze("columns")
#
#    # Evaluar en datos de entrenamiento
#    train_predictions = model.predict(X_train)
#    train_accuracy = accuracy_score(y_train, train_predictions)
#    train_f1 = f1_score(y_train, train_predictions, average='weighted') # weighted para multiclase
#
#    # Evaluar en datos de prueba
#    test_predictions = model.predict(X_test)
#    test_accuracy = accuracy_score(y_test, test_predictions)
#    test_f1 = f1_score(y_test, test_predictions, average='weighted') # weighted para multiclase
#
#    # Guardar métricas de evaluación
#    report_dict = {
#        "metrics": {
#            "train_accuracy": {"value": train_accuracy},
#            "train_f1_score": {"value": train_f1},
#            "test_accuracy": {"value": test_accuracy},
#            "test_f1_score": {"value": test_f1},
#        }
#    }
#
#    # SageMaker espera el archivo de métricas en /opt/ml/output/data/evaluation.json por defecto
#    # Aseguramos la estructura para el Model Quality Report si se usa en el ModelStep
#    evaluation_path = os.path.join(args.output_data_dir, 'evaluation.json')
#    with open(evaluation_path, 'w') as f:
#        json.dump(report_dict, f)
#    print(f"Evaluation metrics saved to {evaluation_path}")


import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import joblib
import os
import json
import tarfile

# Rutas estándar en SageMaker
model_dir = "/opt/ml/processing/model"
input_data_dir = "/opt/ml/processing/input"
output_data_dir = "/opt/ml/processing/output"

# --- EXTRAER MODELO SI ESTÁ COMPRIMIDO ---
tar_path = os.path.join(model_dir, "model.tar.gz")
if os.path.exists(tar_path):
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=model_dir)
    print(f"✅ Modelo extraído desde {tar_path} a {model_dir}")
else:
    print(f"⚠️ No se encontró el archivo model.tar.gz en {model_dir}")

# --- CARGAR MODELO ---
model_path = os.path.join(model_dir, "model.joblib")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"❌ No se encontró el modelo en: {model_path}")
model = joblib.load(model_path)

# --- CARGAR DATOS ---
X_train = pd.read_csv(os.path.join(input_data_dir, 'X_train.csv'))
y_train = pd.read_csv(os.path.join(input_data_dir, 'y_train.csv')).squeeze("columns")
X_test = pd.read_csv(os.path.join(input_data_dir, 'X_test.csv'))
y_test = pd.read_csv(os.path.join(input_data_dir, 'y_test.csv')).squeeze("columns")

# --- PREDICCIONES Y MÉTRICAS ---
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

train_accuracy = accuracy_score(y_train, train_predictions)
train_f1 = f1_score(y_train, train_predictions, average='weighted')

test_accuracy = accuracy_score(y_test, test_predictions)
test_f1 = f1_score(y_test, test_predictions, average='weighted')

# --- GUARDAR REPORTE ---
report_dict = {
    "metrics": {
        "train_accuracy": {"value": train_accuracy},
        "train_f1_score": {"value": train_f1},
        "test_accuracy": {"value": test_accuracy},
        "test_f1_score": {"value": test_f1},
    }
}

evaluation_path = os.path.join(output_data_dir, 'evaluation.json')
with open(evaluation_path, 'w') as f:
    json.dump(report_dict, f)

print(f"✅ Evaluation metrics saved to {evaluation_path}")