import os
import sagemaker
from sagemaker.processing import ScriptProcessor
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.model import Model
from sagemaker.workflow.steps import ProcessingStep, TrainingStep, EvaluationStep
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.functions import JsonGet
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
    ModelQuality,
)
from sagemaker.workflow.parameters import (
    ParameterFloat,
    ParameterString,
)

def get_sagemaker_pipeline(
    region: str,
    role: str,
    default_bucket: str,
    base_job_prefix: str,
    model_package_group_name: str,
    # raw_data_input_uri: str, # No se usará directamente como parámetro S3, el get_data lo genera
    model_accuracy_threshold: float = 0.95, # Umbral de precisión
    sample_rate_for_batch_predict: float = 0.1 # Tasa de muestreo para predicciones batch
) -> Pipeline:

    sagemaker_session = sagemaker.Session(default_bucket=default_bucket)

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
        image_uri=sagemaker.image_uris.get_sklearn_image_uri(region, "1.0-1"),
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-get-data",
        sagemaker_session=sagemaker_session,
    )
    get_data_step = ProcessingStep(
        name="GetData",
        processor=get_data_processor,
        outputs=[sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/raw")],
        code=os.path.join(os.path.dirname(__file__), "../ml_code/get_data.py"),
    )

    # --- Paso 2: Preprocesamiento (Preprocess Data) ---
    preprocess_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.get_sklearn_image_uri(region, "1.0-1"),
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        sagemaker_session=sagemaker_session,
    )
    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[sagemaker.processing.ProcessingInput(
            source=get_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )],
        outputs=[
            sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/processed"),
            sagemaker.processing.ProcessingOutput(source="/opt/ml/model", destination=f"s3://{default_bucket}/iris-artifacts/scaler", output_name="scaler_model") # Para guardar el scaler
        ],
        code=os.path.join(os.path.dirname(__file__), "../ml_code/preprocess.py"),
    )

    # --- Paso 3: Dividir Datos (Split Data) ---
    split_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.get_sklearn_image_uri(region, "1.0-1"),
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-split-data",
        sagemaker_session=sagemaker_session,
    )
    split_data_step = ProcessingStep(
        name="SplitData",
        processor=split_data_processor,
        inputs=[sagemaker.processing.ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )],
        outputs=[
            sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/split")
        ],
        code=os.path.join(os.path.dirname(__file__), "../ml_code/split_data.py"),
    )

    # --- Paso 4: Entrenamiento (Train Model) ---
    sklearn_estimator = SKLearn(
        entry_point=os.path.join(os.path.dirname(__file__), "../ml_code/train_model.py"),
        role=role,
        instance_type="ml.m5.large",
        framework_version="1.0-1",
        base_job_name=f"{base_job_prefix}-train",
        sagemaker_session=sagemaker_session,
        output_path=f"s3://{default_bucket}/iris-artifacts/model", # Donde SageMaker guarda el modelo entrenado
    )
    train_step = TrainingStep(
        name="TrainModel",
        estimator=sklearn_estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=split_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                content_type="text/csv" # Esto es para el canal 'training' en train_model.py
            )
        },
    )

    # --- Paso 5: Evaluación (Evaluate Model) ---
    evaluation_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.get_sklearn_image_uri(region, "1.0-1"),
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluate",
        sagemaker_session=sagemaker_session,
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation", # Este es el nombre de la salida de ProcessingOutput
        source_types=["JSON"],
        path="evaluation.json", # Este es el path dentro del output de /opt/ml/processing/output
    )
    evaluate_step = EvaluationStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts,
                destination="/opt/ml/processing/model",
            ),
            sagemaker.processing.ProcessingInput(
                source=split_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
                destination="/opt/ml/processing/input", # Para X_train, y_train, X_test, y_test
            ),
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-artifacts/evaluation_report")
        ],
        code=os.path.join(os.path.dirname(__file__), "../ml_code/evaluate_model.py"),
        property_files=[evaluation_report],
    )

    # --- Paso 6: Condición de Registro del Modelo (Conditionally Register Model) ---
    # La precisión de prueba debe ser >= umbral
    model_accuracy = JsonGet(
       step_name=evaluate_step.name,
       property_file=evaluation_report,
       json_path="metrics.test_accuracy.value"
    )
    # Condición: Si el umbral (0.95) es MENOR O IGUAL que la precisión del modelo (model_accuracy)
    cond_gt_equal = ConditionLessThanOrEqualTo(left=pipeline_model_accuracy_threshold, right=model_accuracy)

    # Paso para Registrar el Modelo (se ejecuta si la condición se cumple)
    # Crea un ModelPackage para el registro
    model_package_args = Model(
        image_uri=sklearn_estimator.image_uri,
        model_data=train_step.properties.ModelArtifacts.S3ModelArtifacts,
        sagemaker_session=sagemaker_session,
        role=role,
        # Opcional: Registra métricas de evaluación con el modelo.
        # Estas métricas se verán en el Model Registry en la UI.
        model_metrics=ModelMetrics(
            model_quality=ModelQuality(
                statistics=MetricsSource(
                    s3_uri=evaluate_step.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri,
                    content_type="application/json"
                )
            )
        )
    ).register(
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.t2.medium", "ml.m5.large"], # Instancias de inferencia posibles
        transform_instances=["ml.m5.large"], # Instancias de Batch Transform posibles
        model_package_group_name=model_package_group_name,
    )

    register_model_step = ModelStep(
        name="RegisterIrisModel",
        step_args=model_package_args,
    )

    # --- Paso 7: Generar Predicciones Batch (Batch Predict) ---
    # Este paso solo se ejecuta si el modelo se registra exitosamente.
    batch_predict_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.get_sklearn_image_uri(region, "1.0-1"),
        role=role,
        instance_type="ml.m5.large",
        instance_count=1,
        base_job_name=f"{base_job_prefix}-batch-predict",
        sagemaker_session=sagemaker_session,
    )
    batch_predict_step = ProcessingStep(
        name="GenerateBatchPredictions",
        processor=batch_predict_processor,
        inputs=[
            sagemaker.processing.ProcessingInput(
                source=train_step.properties.ModelArtifacts.S3ModelArtifacts, # Para el modelo entrenado
                destination="/opt/ml/processing/model"
            ),
            sagemaker.processing.ProcessingInput(
                source=preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri, # Para datos procesados completos
                destination="/opt/ml/processing/input"
            ),
            sagemaker.processing.ProcessingInput(
                source=f"s3://{default_bucket}/config", # Ruta al prod_config.json que se subirá
                destination="/opt/ml/processing/config"
            )
        ],
        outputs=[
            sagemaker.processing.ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-predictions/batch")
        ],
        code=os.path.join(os.path.dirname(__file__), "../ml_code/batch_predict.py"),
        # Los job_arguments no son necesarios si el script lee del config/prod_config.json
        # job_arguments=[
        #     f"--sample-rate={pipeline_sample_rate_for_batch_predict}"
        # ]
    )

    # Agrupamos los pasos condicionales
    condition_step = ConditionStep(
        name="CheckModelAccuracyAndRegister",
        conditions=[cond_gt_equal],
        if_steps=[register_model_step, batch_predict_step], # Si la condición es True, registra y hace predicción batch
        else_steps=[], # Si es False, no hace nada (o puedes añadir un paso de notificación de falla de umbral)
    )

    # Define el pipeline principal
    pipeline = Pipeline(
        name=f"{base_job_prefix}-IrisMLOpsPipeline",
        parameters=[
            pipeline_model_accuracy_threshold,
            # pipeline_raw_data_input_uri, # Ya no es necesario como parámetro, get_data lo genera
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
    region = os.environ.get("AWS_REGION")
    role = os.environ.get("SAGEMAKER_EXECUTION_ROLE_ARN")
    default_bucket = os.environ.get("S3_ARTIFACT_BUCKET")
    base_job_prefix = os.environ.get("SAGEMAKER_BASE_JOB_PREFIX", "iris-mlops-p")
    model_package_group_name = os.environ.get("SAGEMAKER_MODEL_PACKAGE_GROUP_NAME")
    # raw_data_s3_uri = os.environ.get("RAW_DATA_S3_URI") # Ya no se necesita, get_data lo simula

    # Si se ejecuta localmente para prueba, proveer valores por defecto
    if not all([region, role, default_bucket, model_package_group_name]):
        print("Warning: Missing environment variables. Using dummy values for local testing.")
        region = region or "us-east-1" # Cambia esto a tu región de AWS
        role = role or "arn:aws:iam::123456789012:role/service-role/AmazonSageMaker-ExecutionRole-20230101T123456" # Reemplazar con un ARN válido
        default_bucket = default_bucket or "your-unique-mlops-bucket-12345" # Reemplazar con un nombre de bucket válido
        model_package_group_name = model_package_group_name or "IrisModelGroup"

    pipeline = get_sagemaker_pipeline(
        region=region,
        role=role,
        default_bucket=default_bucket,
        base_job_prefix=base_job_prefix,
        model_package_group_name=model_package_group_name,
        # raw_data_input_uri="dummy_uri" # Ya no es necesario
    )
    
    # Upsert el pipeline (crea o actualiza)
    print(f"Upserting SageMaker Pipeline: {pipeline.name}")
    pipeline.upsert(role_arn=role)

    # Iniciar la ejecución del pipeline (esto se hará desde CodeBuild)
    print(f"Starting SageMaker Pipeline execution for: {pipeline.name}")
    execution = pipeline.start()
    print(f"SageMaker Pipeline execution ARN: {execution.arn}")
    # En CodeBuild, es útil esperar la finalización o monitorear el estado
    # execution.wait() # No lo hacemos aquí para no bloquear CodeBuild si el pipeline es largo
    print("SageMaker Pipeline execution initiated. Check SageMaker Pipelines console for status.")