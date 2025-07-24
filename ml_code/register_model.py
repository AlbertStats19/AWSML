#import argparse
#import os
#import json
#import boto3
#from sagemaker.model import Model
#from sagemaker import image_uris, get_execution_role
#from sagemaker.workflow.parameters import ParameterString # No se usa directamente en el script, pero es bueno tenerlo en mente para el pipeline

#if __name__ == "__main__":
#    parser = argparse.ArgumentParser()
#    # No necesitamos argumentos para model-data, model-package-group-name, region, role-arn directamente
#    # porque los obtendremos de las variables de entorno o de la sesión de SageMaker
#    args = parser.parse_args()
#
#    print("Iniciando el script de registro del modelo...")
#
#    # Obtener variables de entorno de SageMaker Processing Job
#    # Estas son proporcionadas automáticamente por SageMaker
#    model_data_input_path = "/opt/ml/processing/model_data"
#    
#    # SageMaker monta el S3 URI de la entrada bajo esta ruta local
#    # Esperamos que el archivo .tar.gz del modelo esté en esta ubicación
#    model_artifact_path_local = os.path.join(model_data_input_path, "model.tar.gz")
#
#    # Asegúrate de que el archivo del modelo exista
#    if not os.path.exists(model_artifact_path_local):
#        print(f"ERROR: El archivo del modelo no se encontró en {model_artifact_path_local}")
#        # Puedes añadir una lógica para salir o lanzar una excepción si es crítico
#        exit(1) # Salir con un código de error
#    
#    print(f"Model artifact found locally at: {model_artifact_path_local}")
#
#    # Obtener el S3 URI real del artefacto del modelo desde la variable de entorno
#    # SageMaker setea SM_CHANNEL_MODEL_DATA_S3_URI por cada input canalizado
#    model_data_s3_uri = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
#    if not model_data_s3_uri:
#        print("WARNING: SM_CHANNEL_MODEL_DATA_S3_URI no está definido. Intentando construir desde la entrada.")
#        # Fallback si la variable de entorno no está seteada como se espera,
#        print("FALLO CRÍTICO: No se pudo obtener el S3 URI del artefacto del modelo. Es necesario pasarlo como argumento o a través de un canal de entrada específico.")
#        exit(1)
#
#
#    # Usar las variables de entorno de SageMaker para la región y el rol
#    # Si estas no están disponibles, SageMaker Session intentará determinarlas
#    aws_region = os.environ.get("AWS_REGION", "us-east-1") # Fallback a us-east-1
#    boto_session = boto3.Session(region_name=aws_region)
#    sagemaker_session = sagemaker.Session(boto_session=boto_session)
#
#    # El rol se puede obtener del contexto de ejecución si el job tiene permisos
#    # O se puede pasar como un input al script (aunque evitamos argumentos)
#    try:
#        role = get_execution_role(sagemaker_session)
#    except ValueError:
#        print("No se pudo obtener el rol de ejecución de SageMaker. Asegúrate de que el trabajo se esté ejecutando en SageMaker con un rol adecuado.")
#        # Para el propósito de este pipeline, asumiremos que se puede obtener o que la variable de entorno Sagemaker lo tiene.
#        # Alternativamente, puedes pasar el rol como un input al script si quieres evitar el `arguments`
#        print("Intentando obtener rol de la variable de entorno AWS_ROLE_ARN...")
#        role = os.environ.get("AWS_ROLE_ARN")
#        if not role:
#            raise Exception("No se pudo determinar el ARN del rol de IAM. Es requerido para registrar el modelo.")
#        print(f"Usando rol: {role}")
#
#
#    # El S3 URI del artefacto del modelo debe ser conocido aquí.
#    # SageMaker guarda la ruta S3 original de un input en una variable de entorno `SM_CHANNEL_<nombre_input>_S3_URI`.
#    # Así que, para la entrada 'model_data' (definida en el ProcessingStep), la variable será:
#    model_s3_uri_from_env = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
#    if not model_s3_uri_from_env:
#        raise Exception("No se pudo obtener el S3 URI del artefacto del modelo desde la variable de entorno SM_CHANNEL_MODEL_DATA_S3_URI. Asegúrate de que el input 'model_data' esté configurado en el ProcessingStep.")
#    
#    print(f"Obtenido S3 URI del modelo de SM_CHANNEL_MODEL_DATA_S3_URI: {model_s3_uri_from_env}")
#    
#    # El Model Package Group Name NO PUEDE ser inferido. Debe pasarse.
#    # Si `arguments` en ProcessingStep es el problema, la única forma es:
#    #   A) Hardcodearlo (mala práctica para producción)
#    #   B) Leerlo de un archivo de configuración JSON que el pipeline coloque en S3.
#    # Vamos a establecer una entrada para la configuración
#    config_input_path = "/opt/ml/processing/config"
#    config_file_path = os.path.join(config_input_path, "register_config.json")
#
#    # Asegúrate de que el archivo de configuración exista
#    if not os.path.exists(config_file_path):
#        raise FileNotFoundError(f"El archivo de configuración no se encontró en {config_file_path}. El pipeline debe proporcionarlo como input.")
#
#    with open(config_file_path, "r") as f:
#        config = json.load(f)
#
#    model_package_group_name = config.get("model_package_group_name")
#    # La región y el rol se seguirán infiriendo del entorno o de boto3.
#    # Si necesitas una región específica, puedes añadirla al JSON.
#
#    if not model_package_group_name:
#        raise ValueError("model_package_group_name no se encontró en el archivo de configuración.")
#
#    print(f"Obtenido model_package_group_name del archivo de configuración: {model_package_group_name}")
#
#    # Continuar con el S3 URI del modelo como antes
#    model_s3_uri_from_env = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
#    if not model_s3_uri_from_env:
#        raise Exception("No se pudo obtener el S3 URI del artefacto del modelo desde la variable de entorno SM_CHANNEL_MODEL_DATA_S3_URI. Asegúrate de que el input 'model_data' esté configurado en el ProcessingStep.")
#    
#    print(f"Obtenido S3 URI del modelo de SM_CHANNEL_MODEL_DATA_S3_URI: {model_s3_uri_from_env}")
#
#    # Ahora, el registro del modelo con los parámetros obtenidos
#    try:
#        model = Model(
#            image_uri=image_uris.retrieve(framework="sklearn", region=aws_region, version="1.0-1", instance_type="ml.m5.large", image_scope="training"),
#            model_data=model_s3_uri_from_env,
#            role=role,
#            sagemaker_session=sagemaker_session,
#        )
#        print("Modelo SageMaker creado exitosamente.")
#
#        # Registrar el modelo
#        model_package = model.register(
#            content_types=["text/csv"], # Ajusta según tu entrada
#            response_types=["text/csv"], # Ajusta según tu salida
#            inference_instances=["ml.m5.large"], # Instancias para el endpoint
#            transform_instances=["ml.m5.large"], # Instancias para batch transform
#            model_package_group_name=model_package_group_name,
#            # Se añade model_metrics si está disponible (desde evaluate_model.py)
#            # model_metrics=model_metrics # Esto requiere pasar el evaluation.json aquí
#        )
#        print(f"Modelo registrado exitosamente: {model_package.model_package_arn}")
#    except Exception as e:
#        print(f"Error al registrar el modelo: {e}")
#        raise # Re-lanzar la excepción para que el trabajo falle


import joblib
import os
import json
import boto3
from sagemaker import ModelPackage
from sagemaker.session import Session

# Rutas fijas para entradas y config
model_dir = "/opt/ml/processing/model_data/model.joblib"
config_path = "/opt/ml/processing/config/register_config.json"
evaluation_path = "/opt/ml/processing/evaluation/evaluation.json"

# Cargar configuración de registro
with open(config_path, "r") as f:
    config = json.load(f)

model_package_group_name = config["model_package_group_name"]
region = config["region"]
role_arn = config["role_arn"]

# Cargar métricas del modelo
with open(evaluation_path, "r") as f:
    evaluation = json.load(f)

metrics = evaluation.get("metrics", {})
accuracy = metrics.get("test_accuracy", {}).get("value", 0)

# Crear Session y registrar
boto_session = boto3.Session(region_name=region)
sm_session = Session(boto_session=boto_session)

model_package = ModelPackage(
    role=role_arn,
    model_data=model_dir,
    image_uri="683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:1.0-1-cpu-py3",
    sagemaker_session=sm_session
)

model_package.register(
    content_types=["text/csv"],
    response_types=["text/csv"],
    model_package_group_name=model_package_group_name,
    model_metrics={
        "EvaluationMetrics": {
            "Metrics": {
                "test_accuracy": {
                    "Value": accuracy,
                    "StandardMetricName": "Accuracy"
                }
            }
        }
    },
    approval_status="PendingManualApproval"
)
