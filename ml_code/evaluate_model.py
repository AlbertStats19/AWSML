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