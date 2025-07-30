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
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.parameters import ParameterFloat

parameter_custom_lib = "ml.t3.medium"
parameter_train_job = "ml.c5.xlarge"

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

    pipeline_model_accuracy_threshold = ParameterFloat(
        name="ModelAccuracyThreshold",
        default_value=model_accuracy_threshold,
    )
    pipeline_sample_rate_for_batch_predict = ParameterFloat(
        name="SampleRateForBatchPredict",
        default_value=sample_rate_for_batch_predict,
    )

    get_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-get-data",
        sagemaker_session=sagemaker_session,
        command=["python3"],
    )
    get_data_step = ProcessingStep(
        name="GetData",
        processor=get_data_processor,
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/iris-data/raw",
            output_name="output"
        )],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/get_data.py"),
    )

    preprocess_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-preprocess",
        sagemaker_session=sagemaker_session,
        command=["python3"],
    )
    preprocess_step = ProcessingStep(
        name="PreprocessData",
        processor=preprocess_processor,
        inputs=[ProcessingInput(
            source=get_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )],
        outputs=[
            ProcessingOutput(source="/opt/ml/processing/output", destination=f"s3://{default_bucket}/iris-data/processed", output_name="output"),
            ProcessingOutput(source="/opt/ml/processing/scaler", destination=f"s3://{default_bucket}/iris-artifacts/scaler", output_name="scaler_model")
        ],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/preprocess.py"),
    )

    split_data_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-split-data",
        sagemaker_session=sagemaker_session,
        command=["python3"],
    )
    split_data_step = ProcessingStep(
        name="SplitData",
        processor=split_data_processor,
        inputs=[ProcessingInput(
            source=preprocess_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri,
            destination="/opt/ml/processing/input"
        )],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/iris-data/split",
            output_name="output"
        )],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/split_data.py"),
    )

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

    evaluation_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-evaluate",
        sagemaker_session=sagemaker_session,
        command=["python3"],
    )
    evaluation_report = PropertyFile(
        name="EvaluationReport",
        output_name="evaluation",
        path="evaluation.json",
    )
    evaluate_step = ProcessingStep(
        name="EvaluateModel",
        processor=evaluation_processor,
        inputs=[
            ProcessingInput(source=train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=split_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri, destination="/opt/ml/processing/input"),
        ],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/iris-artifacts/evaluation_report",
            output_name="evaluation"
        )],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/evaluate_model.py"),
        property_files=[evaluation_report],
    )

    batch_predict_processor = ScriptProcessor(
        image_uri=sagemaker.image_uris.retrieve(framework="sklearn", region=region, version="1.0-1"),
        role=role,
        instance_type=parameter_custom_lib,
        instance_count=1,
        base_job_name=f"{base_job_prefix}-batch-predict",
        sagemaker_session=sagemaker_session,
        command=["python3"],
    )
    batch_predict_step = ProcessingStep(
        name="GenerateBatchPredictions",
        processor=batch_predict_processor,
        inputs=[
            ProcessingInput(source=train_step.properties.ModelArtifacts.S3ModelArtifacts, destination="/opt/ml/processing/model"),
            ProcessingInput(source=get_data_step.properties.ProcessingOutputConfig.Outputs["output"].S3Output.S3Uri, destination="/opt/ml/processing/input"),
            ProcessingInput(source=f"s3://{default_bucket}/config/prod_config.json", destination="/opt/ml/processing/config"),
        ],
        outputs=[ProcessingOutput(
            source="/opt/ml/processing/output",
            destination=f"s3://{default_bucket}/iris-predictions/batch"
        )],
        code=os.path.join(os.path.dirname(os.path.abspath(__file__)), "../ml_code/batch_predict.py"),
    )

    model_accuracy = JsonGet(
        step_name=evaluate_step.name,
        property_file=evaluation_report,
        json_path="metrics.test_accuracy.value"
    )
    cond_gt_equal = ConditionGreaterThanOrEqualTo(left=model_accuracy, right=pipeline_model_accuracy_threshold)
    condition_step = ConditionStep(
        name="CheckModelAccuracyAndRegister",
        conditions=[cond_gt_equal],
        if_steps=[batch_predict_step],
        else_steps=[],
    )

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
            condition_step
        ],
        sagemaker_session=sagemaker_session,
    )
    return pipeline
