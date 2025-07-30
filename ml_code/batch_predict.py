import os
import tarfile
import pandas as pd
import joblib
import json
import boto3

print("==> Iniciando predicción por lotes...")

# Rutas
compressed_model_path = "/opt/ml/processing/model/model.tar.gz"
extracted_model_path = "/opt/ml/processing/model/model.joblib"
input_data_path = "/opt/ml/processing/input/iris_raw.csv"
batch_config_path = "/opt/ml/processing/config/prod_config.json"
register_config_path = "/opt/ml/processing/config/register_config.json"
output_path = "/opt/ml/processing/output/batch_pred.csv"

# Descomprimir modelo
print("==> Descomprimiendo modelo...")
with tarfile.open(compressed_model_path, "r:gz") as tar:
    tar.extractall(path="/opt/ml/processing/model")

# Cargar modelo
print("==> Cargando modelo descomprimido...")
model = joblib.load(extracted_model_path)

# Cargar datos
df = pd.read_csv(input_data_path)
df = df[[col for col in df.columns if col != "target"]].copy()

# Hacer predicciones
df["prediction"] = model.predict(df)

# Leer configuración de predicción
with open(batch_config_path, "r") as f:
    batch_config = json.load(f)

sample_rate = batch_config["BATCH_PREDICTION"]["SAMPLE_RATE"]
output_file_name = batch_config["BATCH_PREDICTION"]["OUTPUT_FILE_NAME"]
df_sample = df.sample(frac=sample_rate)
df_sample.to_csv(output_path, index=False)
print(f"✅ Predicciones guardadas en: {output_path}")

# --- Registro del modelo usando register_config.json ---
print("==> Iniciando registro del modelo...")

with open(register_config_path, "r") as f:
    register_config = json.load(f)

model_package_group_name = register_config["model_package_group_name"]
region = register_config["region"]
image_uri = register_config["image_uri"]
evaluation_s3_uri = register_config["evaluation_s3_uri"]

# Obtener S3 path dinámico desde variable de entorno
model_data_base_uri = os.environ.get("SM_CHANNEL_MODEL")
if not model_data_base_uri:
    raise EnvironmentError("❌ No se encontró SM_CHANNEL_MODEL")

model_data_url = f"{model_data_base_uri}/model.tar.gz"
print(f"📦 Registrando modelo desde: {model_data_url}")

# Cliente y registro
sm_client = boto3.client("sagemaker", region_name=region)
response = sm_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription="Modelo Iris registrado automáticamente",
    InferenceSpecification={
        "Containers": [
            {
                "Image": image_uri,
                "ModelDataUrl": model_data_url
            }
        ],
        "SupportedContentTypes": ["text/csv"],
        "SupportedResponseMIMETypes": ["text/csv"]
    },
    ModelApprovalStatus="PendingManualApproval",
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": evaluation_s3_uri
            }
        }
    }
)

print("✅ Modelo registrado exitosamente:")
print(f"🔗 ARN: {response['ModelPackageArn']}")
