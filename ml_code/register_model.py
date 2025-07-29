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

# Crear cliente boto3
sm_client = boto3.client("sagemaker", region_name=region)

# Registrar modelo en SageMaker Model Registry
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
