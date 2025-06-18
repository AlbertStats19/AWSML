# Plantilla MLOps de Madurez III para Clasificación de Iris en AWS

Este repositorio contiene una plantilla de ejemplo para implementar un pipeline MLOps de nivel de madurez III en AWS, utilizando el dataset de Iris como caso de estudio.

## Características:
- **Automatización Completa:** Desencadenado por commits en el repositorio de código.
- **SageMaker Pipelines:** Orquestación de pasos de ML (preprocesamiento, entrenamiento, evaluación, registro de modelo).
- **Control de Calidad del Modelo:** Condición de registro del modelo basada en métricas de evaluación (ej. `accuracy >= 0.95`).
- **Generación de Predicciones Batch:** Si el modelo cumple la condición de calidad, se genera un archivo de predicciones en lote.
- **Contenerización:** Las tareas de ML se ejecutan en contenedores de SageMaker.
- **Infraestructura como Código (IaC):** Despliegue de todos los recursos de AWS usando CloudFormation.
- **Notificaciones:** Configuración para recibir alertas por correo electrónico sobre el estado del pipeline.

## Servicios de AWS Utilizados:
- **AWS CodeCommit:** Repositorio de código fuente.
- **AWS CodePipeline:** Orquestador del pipeline CI/CD.
- **AWS CodeBuild:** Para ejecutar tareas de construcción, pruebas y orquestar SageMaker Pipelines.
- **Amazon S3:** Almacenamiento de datos, artefactos y modelos.
- **AWS SageMaker:**
    - **SageMaker Pipelines:** Para definir y ejecutar el flujo de trabajo de ML.
    - **SageMaker Processing Jobs:** Para preprocesamiento y evaluación.
    - **SageMaker Training Jobs:** Para el entrenamiento del modelo.
    - **SageMaker Model Registry:** Para versionar y gestionar modelos.
- **Amazon SNS:** Para notificaciones por correo electrónico.
- **Amazon CloudWatch Events (EventBridge):** Para detectar eventos del pipeline y activar notificaciones.
- **Amazon CloudWatch:** Para logs y monitoreo.

## Flujo del Pipeline (Alto Nivel):
1.  **Commit/Push:** Un `git push` a la rama configurada (ej. `main`) en CodeCommit.
2.  **CodePipeline Source:** CodePipeline detecta el cambio y obtiene el código.
3.  **CodeBuild (Build & Train):**
    * Ejecuta `buildspec.yml`.
    * Instala dependencias.
    * Ejecuta pruebas de código (opcional, pero buena práctica).
    * **Activa un SageMaker Pipeline.**
        * Este pipeline:
            * Obtiene y Preprocesa datos (Iris).
            * Divide los datos en entrenamiento/prueba.
            * Entrena el modelo (Logistic Regression).
            * Evalúa el modelo y guarda métricas (accuracy de train/test).
            * **Si `accuracy_test >= 0.95`:**
                * Registra el modelo en SageMaker Model Registry.
                * Genera un archivo `batch_pred.csv` con predicciones de muestra.
    * **Notificación a EventBridge/SNS (configurada en la infraestructura) al finalizar el CodePipeline.**

## Configuración y Despliegue:
Sigue los pasos detallados que te proporcionará tu guía para configurar tu cuenta de AWS y desplegar esta solución.

## Limpieza:
Es crucial limpiar todos los recursos de AWS después de finalizar para evitar costos. Se proporcionarán instrucciones detalladas.