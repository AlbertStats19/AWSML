import json
import boto3

# Cargar configuración
with open("/opt/ml/processing/config/register_config.json", "r") as f:
    config = json.load(f)

model_package_group_name = config["model_package_group_name"]
region = config["region"]
image_uri = config["image_uri"]
model_data_url = config["model_data_url"]
evaluation_s3_uri = config["evaluation_s3_uri"]

## Crear cliente boto3
sm_client = boto3.client("sagemaker", region_name=region)

## Registrar modelo en SageMaker Model Registry
response = sm_client.create_model_package(
    ModelPackageGroupName=model_package_group_name,
    ModelPackageDescription="Modelo Iris con métrica",
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
    ModelApprovalStatus="PendingManualApproval",  # ✅ Este es el correcto
    ModelMetrics={
        "ModelQuality": {
            "Statistics": {
                "ContentType": "application/json",
                "S3Uri": evaluation_s3_uri
            }
        }
    }
)

print("Modelo registrado exitosamente:")
print(response["ModelPackageArn"])


#import os
#import json
#import boto3

# --- Leer configuración ---
#with open("/opt/ml/processing/config/register_config.json", "r") as f:
#    config = json.load(f)

#model_package_group_name = config["model_package_group_name"]
#region = config["region"]
#image_uri = config["image_uri"]
#evaluation_s3_uri = config["evaluation_s3_uri"]

# --- Detectar ruta dinámica del modelo ---
#model_dir = "/opt/ml/processing/model_data"
#model_file_name = "model.tar.gz"
#model_local_path = os.path.join(model_dir, model_file_name)
#
#if not os.path.exists(model_local_path):
#    raise FileNotFoundError(f"❌ No se encontró el modelo en: {model_local_path}")
#
# ✅ Obtener S3 original del modelo desde variable de entorno
#model_data_base_uri = os.environ.get("SM_CHANNEL_MODEL_DATA")

#if not model_data_base_uri:
#    raise EnvironmentError("❌ No se pudo obtener SM_CHANNEL_MODEL_DATA")
#
# Ruta completa en S3 al .tar.gz
#model_data_url = f"{model_data_base_uri}/{model_file_name}"
#
#print(f"📦 ModelDataUrl detectado dinámicamente: {model_data_url}")
#
# --- Registrar el modelo en SageMaker Model Registry ---
#sm_client = boto3.client("sagemaker", region_name=region)
#
#response = sm_client.create_model_package(
#    ModelPackageGroupName=model_package_group_name,
#    ModelPackageDescription="Modelo Iris con métricas evaluadas",
#    InferenceSpecification={
#        "Containers": [
#            {
#                "Image": image_uri,
#                "ModelDataUrl": model_data_url
#            }
#        ],
#        "SupportedContentTypes": ["text/csv"],
#        "SupportedResponseMIMETypes": ["text/csv"]
#    },
#    ModelApprovalStatus="PendingManualApproval",
#    ModelMetrics={
#        "ModelQuality": {
#            "Statistics": {
#                "ContentType": "application/json",
#                "S3Uri": evaluation_s3_uri
#            }
#        }
#    }
#)
#
#print("✅ Modelo registrado exitosamente:")
#print(f"🔗 ARN: {response['ModelPackageArn']}")
