import os
import sagemaker
import json
import boto3
from sagemaker.processing import ScriptProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.steps import ProcessingStep, TrainingStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
#from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterString,
)

# parameter_custom_lib = "ml.m5.large"
parameter_custom_lib = "ml.t3.medium"
parameter_train_job = "ml.c5.xlarge"


# La función get_sagemaker_pipeline ahora aceptará los parámetros
# que serán inyectados por CodeBuild.
def get_sagemaker_pipeline(
    region: str,
    role: str,
    default_bucket: str,
    base_job_prefix: str,
    model_package_group_name: str,
    model_accuracy_threshold: float = 0.90,
    sample_rate_for_batch_predict: float = 0.1
) -> Pipeline:

    sagemaker_session = sagemaker.Session(default_bucket=default_bucket)
    boto_session = boto3.Session(region_name=region) # Para S3 client

    # Parámetros del Pipeline (pueden ser modificados al iniciar el pipeline)
    pipeline_model_accuracy_threshold = ParameterFloat(
        name="ModelAccuracyThreshold",
        default_value=model_accuracy_threshold,
    )
    pipeline_sample_rate_for_batch_predict = ParameterFloat(
        name="SampleRateForBatchPredict",
        default_value=sample_rate_for_batch_predict,
    )

    # --- Paso 1: Obtener Datos (Get Data) ---
    get_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-get-data",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )
    #get_data_step = ProcessingStep(
    #    name="GetData",
    #    processor=get_data_processor,
    #    outputs=[ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/raw")],
    #    # La ruta del código debe ser relativa al directorio donde se ejecuta el pipeline.py
    #    # Cuando CodeBuild ejecute este archivo, CodeBuild estará en el directorio raíz del repo.
    #    code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/get_data.py"),
    #)

    get_data_step = ProcessingStep(
        name="GetData",
        processor=get_data_processor,
        outputs=[
        ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/iris-data/raw",
            output_name="output"   # ✅ ESTE NOMBRE ES CLAVE
        )
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/get_data.py"),
    )

    # --- Paso 2: Preprocesamiento (Preprocess Data) ---
    preprocess_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )
    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[ProcessingInput(
            source=get_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input",
            # input_name="input_data"
        )],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/processed",  output_name="output" ),
            # ProcessingOutput(source="/opt/ml/model", destination=f"s3://{default_bucket}/iris-artifacts/scaler", output_name="scaler_model")
            ProcessingOutput(source="/opt/ml/processing/scaler", destination=f"s3://{default_bucket}/iris-artifacts/scaler", output_name="scaler_model")
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/preprocess.py"),
    )

    # --- Paso 3: Dividir Datos (Split Data) ---
    split_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-split-data",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )
    split_data_step = ProcessingStep(
        name="SplitData",
        processor=split_data_processor,
        inputs=[ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/split",output_name="output")
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/split_data.py"),
    )

    # --- Paso 4: Entrenamiento (Train Model) ---
    sklearn_estimator = SKLearn(
        entry_point=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/train_model.py"),
        role=role,
        instance_type=parameter_train_job,
        framework_version="1.0-1",
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{default_bucket}/iris-artifacts/model",
        use_spot_instances=True,
        max_wait=3600,
        max_run=1800
    )
    train_step = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=split_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                content_type="text/csv"
            )
        },
    )

    # --- Paso 5: Evaluación (Evaluate Model) ---
    evaluation_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluate",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )
    
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation", # Corregido para que coincida con el nombre del output
        path="evaluation.json",
    )

    evaluate_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            ProcessingInput(
                source=split_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input",
            ),
        ],
        outputs=[
            ProcessingOutput(
                source="/opt/ml/processing/output", 
                destination=f"s3://{default_bucket}/iris-artifacts/evaluation_report",
                output_name="evaluation" # Nombre del output para referenciar con PropertyFile
            )
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/evaluate_model.py"),
        property_files=[evaluation_report],
    )

    # --- Preparación para el Paso de Registro del Modelo (Configuración JSON) ---
    # Crearemos un archivo JSON temporal con la configuración para el registro
    # y lo subiremos a S3 para que el script de registro lo lea.
    # Esto es crucial para pasar `model_package_group_name` sin usar `arguments`.
    
    register_config_data = {
        "model_package_group_name": model_package_group_name,
        "region": region,
        "role_arn": role,
        "image_uri": sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        "evaluation_s3_uri": f"s3://{default_bucket}/iris-artifacts/evaluation_report/evaluation.json"
    }

    #register_config_data = {
    #    "model_package_group_name": model_package_group_name,
    #    "region": region,
    #    "role_arn": role,
    #    "image_uri": sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
    #    "model_data_url": f"s3://{default_bucket}/iris-artifacts/model/model.tar.gz",
    #    "evaluation_s3_uri": f"s3://{default_bucket}/iris-artifacts/evaluation_report/evaluation.json"
    #}

    # MODIFICAR BATCH
    # Editar bucket quemado

    #register_config_data = {
    #    "model_package_group_name": model_package_group_name,
    #    "region": region,
    #    "role_arn": role,
    #    "image_uri": sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
    #    "model_data_url": train_step.properties.ModelArtifacts.S3ModelArtifacts.to_string(),
    #    "evaluation_s3_uri": f"s3://{default_bucket}/iris-artifacts/evaluation_report/evaluation.json"
    #}


    # Ruta local para el archivo de configuración temporal
    # Importante: Esto se ejecutará en CodeBuild, así que la ruta temporal debe ser dentro del entorno de CodeBuild.
    # No es necesario crear un directorio si solo vamos a escribir un archivo.
    register_config_file_name = "register_config.json"
    # Para CodeBuild, la ruta temporal debería ser en /tmp o en el directorio de trabajo del build.
    # Aquí simulamos la creación del archivo localmente para la subida.
    temp_config_path = os.path.join(os.getcwd(), register_config_file_name) 

    with open(temp_config_path, "w") as f:
        json.dump(register_config_data, f)
    
    # Subir el archivo de configuración a S3
    s3_client = boto_session.client("s3")
    s3_config_key = f"pipeline-configs/{base_job_prefix}/{register_config_file_name}"
    s3_config_uri = f"s3://{default_bucket}/{s3_config_key}"
    
    # Este 'upload_file' se ejecutará cuando CodeBuild corra este pipeline.py
    s3_client.upload_file(temp_config_path, default_bucket, s3_config_key)
    print(f"Archivo de configuración de registro subido a S3: {s3_config_uri}")

    # --- Paso 6: REGISTRO DEL MODELO ---
    # Usaremos ScriptProcessor para el registro, y pasaremos la configuración via un S3Input
    register_model_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-register-model",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )

    register_model_step = ProcessingStep(
        name="RegisterModel",
        processor=register_model_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model_data",
                input_name="model_data" # Nombre del canal para SM_CHANNEL_MODEL_DATA_S3_URI
            ),
            ProcessingInput( # <-- NUEVA ENTRADA para el archivo de configuración JSON
                source=s3_config_uri,
                destination="/opt/ml/processing/config",
                input_name="config" # Nombre del canal para SM_CHANNEL_CONFIG_S3_URI
            ),
            ProcessingInput( # <-- ENTRADA para el reporte de evaluación
                source=evaluate_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                destination="/opt/ml/processing/evaluation",
                input_name="evaluation" # Nombre del canal para SM_CHANNEL_EVALUATION_S3_URI
            )
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/register_model.py"),
    )
    
    # --- Paso 7: Generar Predicciones Batch (Batch Predict) ---
    # Este paso se ejecutará si la condición del modelo es verdadera.
    batch_predict_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-batch-predict",
        sagemaker_session=sagemaker_session,
        command=["python3"], # <--- CORRECCIÓN CLAVE
    )
    batch_predict_step = ProcessingStep(
        name="GenerateBatchPredictions",
        processor=batch_predict_processor,
        inputs=[
            ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts, # Para el modelo entrenado
                destination="/opt/ml/processing/model"
            ),
            ProcessingInput(
                source=get_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri, # Para datos procesados completos
                destination="/opt/ml/processing/input"
            ),
            # Asumiendo que el script de batch_predict.py también necesita una configuración
            # Aquí la ruta sería al 'config' general, o a un 'batch_predict_config.json' si lo necesitaras.
            # Por ahora, mantendré la referencia a un S3 config genérico como en tu original.
            # Si el script batch_predict.py no necesita este config, se puede quitar.
            ProcessingInput(
                source=f"s3://{default_bucket}/config/prod_config.json", # Ruta a un archivo de configuración para predicción
                destination="/opt/ml/processing/config"
            )
        ],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-predictions/batch")
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/batch_predict.py"),
    )

    # Agrupamos los pasos condicionales
    model_accuracy = JsonGet(
        step_name=evaluate_step.name,
        property_file=evaluation_report,
        json_path="metrics.test_accuracy.value"
    )
    cond_gt_equal =  ConditionGreaterThanOrEqualTo(left=model_accuracy,right=pipeline_model_accuracy_threshold)

    condition_step = ConditionStep(
        name="CheckModelAccuracyAndRegister",
        conditions=[cond_gt_equal],
        if_steps=[register_model_step, batch_predict_step], # Ambos pasos se ejecutan si la condición es verdadera
        else_steps=[],
    )

    # Define el pipeline principal
    pipeline = Pipeline(
        name=f"{base_job_prefix}-IrisMLOpsPipeline",
        parameters=[
            pipeline_model_accuracy_threshold,
            pipeline_sample_rate_for_batch_predict
        ],
        steps=[
            get_data_step,
            preprocess_step,
            split_data_step,
            train_step,
            evaluate_step,
            condition_step # Agrega el paso condicional
        ],
        sagemaker_session=sagemaker_session,
    )
    return pipeline

if __name__ == "__main__":
    # Estas variables serán pasadas por el entorno de CodeBuild
    # Es crucial que CodeBuild configure estas variables de entorno.
    region = os.environ.get("AWS_REGION")
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN") # ARN del rol que asumirá SageMaker Pipelines
    default_bucket = os.environ.get("S3_ARTIFACT_BUCKET")
    base_job_prefix = os.environ.get("SAGEMAKER_BASE_JOB_PREFIX", "iris-mlops-p")
    model_package_group_name = os.environ.get("SAGEMAKER_MODEL_PACKAGE_GROUP_NAME")
    
    # Bloque de seguridad: Si falta alguna variable de entorno necesaria, no procede.
    # En un entorno CI/CD, estas siempre deberían estar definidas.
    if not all([region, role, default_bucket, model_package_group_name]):
        print("ERROR: Faltan variables de entorno cruciales para el despliegue del pipeline.")
        print(f"AWS_REGION: {region}")
        print(f"SAGEMAKER_EXECUTION_ROLE_ARN: {role}")
        print(f"S3_ARTIFACT_BUCKET: {default_bucket}")
        print(f"SAGEMAKER_MODEL_PACKAGE_GROUP_NAME: {model_package_group_name}")
        # En un script de CodeBuild, un sys.exit(1) detendrá el build.
        import sys
        sys.exit(1) 

    pipeline = get_sagemaker_pipeline(
        region=region,
        role=role,
        default_bucket=default_bucket,
        base_job_prefix=base_job_prefix,
        model_package_group_name=model_package_group_name,
    )
    
    print(f"\n Bucket en uso por este pipeline en AWS: {default_bucket}\n")  # ESTA ES LA LÍNEA CLAVE

    # Upsert el pipeline (crea o actualiza la definición del pipeline en SageMaker)
    print(f"Upserting SageMaker Pipeline: {pipeline.name}")
    # El role_arn en upsert es el rol que el pipeline usará para llamar a otros servicios.
    pipeline.upsert(role_arn=role) 

    # IMPORTANTE: En un pipeline de CI/CD, usualmente NO inicias la ejecución del SageMaker Pipeline
    # inmediatamente después de su upsert.
    # La ejecución del pipeline de ML (SageMaker Pipeline) se suele disparar:
    # 1. Manualmente desde la consola.
    # 2. Por un evento de CloudWatch (ej. carga de nuevos datos a S3).
    # 3. Como un paso posterior en el CodePipeline si lo quieres encadenar.
    #
    # Si quieres que CodeBuild inicie una ejecución *automáticamente* cada vez que se actualiza el pipeline,
    # entonces mantén las siguientes líneas. Si no, coméntalas.
    # Por ahora, las dejo comentadas para darte la opción, ya que el patrón común es no disparar inmediatamente.
    #
    # print(f"Starting SageMaker Pipeline execution for: {pipeline.name}")
    # execution = pipeline.start()
    # print(f"SageMaker Pipeline execution ARN: {execution.arn}")
    # print("SageMaker Pipeline execution initiated. Check SageMaker Pipelines console for status.")