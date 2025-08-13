import pandas as pd
from sklearn.linear_model import LogisticRegression
import joblib
import os

# Rutas estándar en SageMaker
# input_dir = "/opt/ml/processing/input"
input_dir = "/opt/ml/input/data/training"
model_dir = "/opt/ml/model"

# Cargar datos de entrenamiento
X_train_path = os.path.join(input_dir, 'X_train.csv')
y_train_path = os.path.join(input_dir, 'y_train.csv')

X_train = pd.read_csv(X_train_path)
y_train = pd.read_csv(y_train_path).squeeze("columns")  # Squeeze para convertir DataFrame en Series

# Entrenar el modelo
model = LogisticRegression(random_state=42, max_iter=1000)  # Asegurar convergencia
model.fit(X_train, y_train)

# Guardar el modelo entrenado
model_path = os.path.join(model_dir, "model.joblib")
joblib.dump(model, model_path)

print(f"✅ Model trained and saved at: {model_path}")