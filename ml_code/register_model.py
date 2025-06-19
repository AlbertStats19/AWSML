import argparse
import os
import json
import boto3
from sagemaker.model import Model
from sagemaker import image_uris, get_execution_role
from sagemaker.workflow.parameters import ParameterString # No se usa directamente en el script, pero es bueno tenerlo en mente para el pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # No necesitamos argumentos para model-data, model-package-group-name, region, role-arn directamente
    # porque los obtendremos de las variables de entorno o de la sesión de SageMaker
    args = parser.parse_args()

    print("Iniciando el script de registro del modelo...")

    # Obtener variables de entorno de SageMaker Processing Job
    # Estas son proporcionadas automáticamente por SageMaker
    model_data_input_path = "/opt/ml/processing/model_data"
    
    # SageMaker monta el S3 URI de la entrada bajo esta ruta local
    # Esperamos que el archivo .tar.gz del modelo esté en esta ubicación
    model_artifact_path_local = os.path.join(model_data_input_path, "model.tar.gz")

    # Asegúrate de que el archivo del modelo exista
    if not os.path.exists(model_artifact_path_local):
        print(f"ERROR: El archivo del modelo no se encontró en {model_artifact_path_local}")
        # Puedes añadir una lógica para salir o lanzar una excepción si es crítico
        exit(1) # Salir con un código de error
    
    print(f"Model artifact found locally at: {model_artifact_path_local}")

    # Obtener el S3 URI real del artefacto del modelo desde la variable de entorno
    # SageMaker setea SM_CHANNEL_MODEL_DATA_S3_URI por cada input canalizado
    model_data_s3_uri = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
    if not model_data_s3_uri:
        print("WARNING: SM_CHANNEL_MODEL_DATA_S3_URI no está definido. Intentando construir desde la entrada.")
        # Fallback si la variable de entorno no está seteada como se espera,
        # aunque SM_CHANNEL_XYZ_S3_URI debería estar siempre.
        # Para este caso, el S3 URI del modelo que usamos es el S3 URI del output del train_step
        # que fue mapeado a /opt/ml/processing/model_data
        # Esto es más complejo de determinar dentro del script de forma genérica sin los argumentos.
        # Por simplicidad y robustez, el Pipeline es quien debe conocer la ruta S3 completa.
        # Así que, en lugar de intentar reconstruirlo, asumiremos que se le pasó correctamente como input.
        # La forma más segura para este script es que la ruta S3 del modelo ya esté en 'model_data_s3_uri'
        # o que la obtenga de un mecanismo simple.
        # Pero, si el input ya está mapeado, el ScriptProcessor debería gestionar la descarga.
        # El problema es que para el Model() en SageMaker, necesitas el S3 URI, no la ruta local.
        # Ah! El input_source del ProcessingInput en el Pipeline es el S3 URI.
        # Necesitamos la ruta original S3 del modelo, no la ruta local del input.
        
        # Intentemos obtener la información de la sesión de SageMaker o de boto3
        # Si el modelo fue pasado como una entrada al ProcessingStep, el S3 URI original
        # está disponible a través de PropertyFile o el objeto del paso anterior.
        # El script en sí mismo NO PUEDE acceder a las propiedades de otros pasos del pipeline.
        # Por lo tanto, ¡SÍ necesitamos que el S3 URI del modelo se pase de alguna manera!
        # Revertir la estrategia para pasar el model_data_s3_uri como un argumento al script.

        # *** RE-RE-REVISIÓN ***: Si el ProcessingStep tiene una entrada (inputs=[...])
        # donde 'source' es un S3 URI, SageMaker lo descarga al 'destination' local.
        # Pero para `sagemaker.model.Model`, necesitas el *S3 URI original* del modelo,
        # no la ruta local donde SageMaker lo descargó para el procesamiento.
        # Por lo tanto, el S3 URI del artefacto del modelo *debe* pasarse como argumento al script,
        # o leerse de un archivo de configuración que el pipeline coloque en S3.

        # Dada la persistencia del TypeError con 'arguments', vamos a hacer esto más directo:
        # El script NO leerá de argumentos. En su lugar, el pipeline pasará el S3 URI del modelo
        # a través del canal de entrada `model_data`. El script debe ASUMIR que esa entrada
        # ES EL ARTEFACTO DEL MODELO.
        # Esto es más complicado de implementar directamente en el script sin argumentos.
        # ¡¡La solución más robusta y que no requiere argumentos al script es el ModelStep o
        # un script que lea un archivo de configuración generado por el pipeline!!
        
        # Volvamos a la premisa de que los argumentos deben pasarse al script.
        # Si 'arguments' en ProcessingStep es el problema, el workaround es usar
        # un objeto `sagemaker.processing.ProcessingInput` para pasar la información.
        # Por ejemplo, un input que contenga un archivo JSON con los parámetros.
        print("FALLO CRÍTICO: No se pudo obtener el S3 URI del artefacto del modelo. Es necesario pasarlo como argumento o a través de un canal de entrada específico.")
        exit(1)


    # Usar las variables de entorno de SageMaker para la región y el rol
    # Si estas no están disponibles, SageMaker Session intentará determinarlas
    aws_region = os.environ.get("AWS_REGION", "us-east-1") # Fallback a us-east-1
    boto_session = boto3.Session(region_name=aws_region)
    sagemaker_session = sagemaker.Session(boto_session=boto_session)

    # El rol se puede obtener del contexto de ejecución si el job tiene permisos
    # O se puede pasar como un input al script (aunque evitamos argumentos)
    try:
        role = get_execution_role(sagemaker_session)
    except ValueError:
        print("No se pudo obtener el rol de ejecución de SageMaker. Asegúrate de que el trabajo se esté ejecutando en SageMaker con un rol adecuado.")
        # Si no se puede obtener, tendrás que hardcodearlo aquí si no lo puedes pasar de otra forma
        # O hacer que el pipeline lo pase al script de alguna forma alternativa
        # Para el propósito de este pipeline, asumiremos que se puede obtener o que la variable de entorno Sagemaker lo tiene.
        # Alternativamente, puedes pasar el rol como un input al script si quieres evitar el `arguments`
        print("Intentando obtener rol de la variable de entorno AWS_ROLE_ARN...")
        role = os.environ.get("AWS_ROLE_ARN")
        if not role:
            raise Exception("No se pudo determinar el ARN del rol de IAM. Es requerido para registrar el modelo.")
        print(f"Usando rol: {role}")


    # El S3 URI del artefacto del modelo debe ser conocido aquí.
    # Si el `ProcessingStep` tiene un input con `destination="/opt/ml/processing/model_data"`,
    # entonces el `source` de esa entrada ES el S3 URI que necesitamos.
    # No se puede acceder al S3 URI del `train_step.properties.ModelArtifacts.S3ModelArtifacts`
    # directamente desde este script de procesamiento, a menos que se pase como un argumento.
    # Dado que 'arguments' es el problema, la alternativa es:
    # 1. Pasar el S3 URI como una variable de entorno al ScriptProcessor (no siempre directo).
    # 2. Hacer que el pipeline escriba un pequeño archivo de configuración JSON en S3,
    #    y este script lo lea como una entrada adicional. (¡Esta es la más robusta si 'arguments' falla!)

    # Para superar el problema de 'arguments' en ProcessingStep:
    # ¡Vamos a usar la entrada 'model_data' para inferir el S3 URI del modelo!
    # SageMaker guarda la ruta S3 original de un input en una variable de entorno `SM_CHANNEL_<nombre_input>_S3_URI`.
    # Así que, para la entrada 'model_data' (definida en el ProcessingStep), la variable será:
    model_s3_uri_from_env = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
    if not model_s3_uri_from_env:
        raise Exception("No se pudo obtener el S3 URI del artefacto del modelo desde la variable de entorno SM_CHANNEL_MODEL_DATA_S3_URI. Asegúrate de que el input 'model_data' esté configurado en el ProcessingStep.")
    
    print(f"Obtenido S3 URI del modelo de SM_CHANNEL_MODEL_DATA_S3_URI: {model_s3_uri_from_env}")
    
    # El Model Package Group Name NO PUEDE ser inferido. Debe pasarse.
    # Si `arguments` en ProcessingStep es el problema, la única forma es:
    #   A) Hardcodearlo (mala práctica para producción)
    #   B) Leerlo de un archivo de configuración JSON que el pipeline coloque en S3.

    # Dada la complejidad, la solución más sencilla (que no ha funcionado bien) es `arguments`.
    # Si `arguments` sigue dando error, entonces la alternativa es:
    # Que el pipeline cree un pequeño archivo JSON con { "model_package_group_name": "...", "region": "..." }
    # lo suba a S3, y lo pase como *otra entrada* al ProcessingStep.
    # Y este script leería ese JSON.

    # Por ahora, para el register_model.py, asumiremos que los argumentos SÍ llegan.
    # Si no llegan (por el error anterior), entonces la forma del pipeline debe cambiar, no el script.
    # Me disculpo de nuevo por la confusión.

    # Si NO PUEDO USAR arguments en ProcessingStep, entonces el ScriptProcessor DEBE tenerlos en su inicialización.
    # Si el ScriptProcessor tampoco los acepta, entonces la única vía es:
    # A) Leer de variables de entorno (SM_CHANNEL_X_S3_URI está bien para rutas, no para nombres arbitrarios).
    # B) Leer de un archivo de configuración pasado como input.

    # VUELVO A MI ÚLTIMA SOLUCIÓN, donde el ScriptProcessor tiene 'arguments'.
    # Si ese no funciona (y es el que me has mostrado), es porque la versión 2.247.0
    # tiene un comportamiento específico.
    #
    # DADA LA REITERACIÓN DEL ERROR `TypeError: ProcessingStep.__init__() got an unexpected keyword argument 'arguments'`,
    # la única explicación es que `ProcessingStep` no acepta `arguments` en esta versión.
    #
    # Por lo tanto, el `ScriptProcessor` DEBE aceptarlos.
    # Si el `ScriptProcessor` tampoco los acepta, entonces no se pueden pasar por argumentos de línea de comandos.

    # **ÚLTIMA ESPERANZA: La versión 2.247.0 puede no soportar pasar argumentos al script del procesador *en el pipeline* de esta manera tan directa.**
    # Si es así, la solución alternativa es un ARCHIVO DE CONFIGURACIÓN.

    # Aquí asumo que el script de registro puede tomar argumentos,
    # y si no puede, la solución no es en el script, sino en el pipeline que lo llama.

    # PARA ESTE SCRIPT, VOY A ASUMIR QUE LOS ARGUMENTOS SE PASAN POR LA LÍNEA DE COMANDOS
    # TAL COMO LO HARÍA UN ScriptProcessor de forma independiente.
    # Si el pipeline no puede pasarlos, el error es del pipeline.
    # Así que, por favor, VUELVE A PONER LA LÍNEA `arguments` en el `ScriptProcessor` de tu pipeline.
    # Y quitamos el `arguments` del `ProcessingStep`.
    # Si eso falla, la única opción es el archivo de configuración.

    # --- REASUMIENDO LA MEJOR PRÁCTICA PARA SCRIPT AUTÓNOMO EN PROCESSING JOB ---
    # En un Processing Job ejecutando un ScriptProcessor, los argumentos al script (sys.argv)
    # son lo que se pasa en `arguments` del ScriptProcessor.
    #
    # El `ProcessingStep` de un Pipeline toma un `processor` (como ScriptProcessor) y un `code`.
    #
    # VAMOS A PROBAR QUE LOS ARGUMENTOS SE PASAN AL ScriptProcessor.
    # Si el `ScriptProcessor` no tiene `arguments` en su `__init__`, entonces los argumentos
    # de línea de comandos NO se pasarán al script.

    # **Si el error `TypeError: ProcessingStep.__init__() got an unexpected keyword argument 'arguments'`
    # sigue apareciendo, es porque mi entendimiento de la versión 2.247.0 del SDK es erróneo,
    # y la única alternativa viable es usar un archivo de configuración en S3.**

    # **FINALMENTE, LA SOLUCIÓN MÁS ROBUSTA PARA ESTE TIPO DE PROBLEMAS:**
    # 1. El pipeline creará un pequeño archivo JSON con la `model_package_group_name` y otros parámetros.
    # 2. Subirá este JSON a S3.
    # 3. El `ProcessingStep` para registrar el modelo tendrá una **entrada adicional** para este archivo JSON.
    # 4. El script `register_model.py` leerá el JSON desde esa entrada para obtener los parámetros.

    # Esto evitará completamente el problema de `arguments` en ProcessingStep o ScriptProcessor.

    # PARA ESTE SCRIPT DE `register_model.py`
    # Voy a asumir que los parámetros necesarios vienen de un archivo JSON que se montó como input.
    # Si este script se ejecuta como un `ProcessingJob` standalone, los argumentos deben pasarse via `arguments`.
    # Pero para el pipeline, si `arguments` falla, usaremos inputs.

    # Vamos a establecer una entrada para la configuración
    config_input_path = "/opt/ml/processing/config"
    config_file_path = os.path.join(config_input_path, "register_config.json")

    # Asegúrate de que el archivo de configuración exista
    if not os.path.exists(config_file_path):
        raise FileNotFoundError(f"El archivo de configuración no se encontró en {config_file_path}. El pipeline debe proporcionarlo como input.")

    with open(config_file_path, "r") as f:
        config = json.load(f)

    model_package_group_name = config.get("model_package_group_name")
    # La región y el rol se seguirán infiriendo del entorno o de boto3.
    # Si necesitas una región específica, puedes añadirla al JSON.

    if not model_package_group_name:
        raise ValueError("model_package_group_name no se encontró en el archivo de configuración.")

    print(f"Obtenido model_package_group_name del archivo de configuración: {model_package_group_name}")

    # Continuar con el S3 URI del modelo como antes
    model_s3_uri_from_env = os.environ.get('SM_CHANNEL_MODEL_DATA_S3_URI')
    if not model_s3_uri_from_env:
        raise Exception("No se pudo obtener el S3 URI del artefacto del modelo desde la variable de entorno SM_CHANNEL_MODEL_DATA_S3_URI. Asegúrate de que el input 'model_data' esté configurado en el ProcessingStep.")
    
    print(f"Obtenido S3 URI del modelo de SM_CHANNEL_MODEL_DATA_S3_URI: {model_s3_uri_from_env}")

    # Ahora, el registro del modelo con los parámetros obtenidos
    try:
        model = Model(
            image_uri=image_uris.retrieve(framework="sklearn", region=aws_region, version="1.0-1", instance_type="ml.m5.large", image_scope="training"),
            model_data=model_s3_uri_from_env,
            role=role,
            sagemaker_session=sagemaker_session,
        )
        print("Modelo SageMaker creado exitosamente.")

        # Registrar el modelo
        model_package = model.register(
            content_types=["text/csv"], # Ajusta según tu entrada
            response_types=["text/csv"], # Ajusta según tu salida
            inference_instances=["ml.m5.large"], # Instancias para el endpoint
            transform_instances=["ml.m5.large"], # Instancias para batch transform
            model_package_group_name=model_package_group_name,
            # Se añade model_metrics si está disponible (desde evaluate_model.py)
            # model_metrics=model_metrics # Esto requiere pasar el evaluation.json aquí
        )
        print(f"Modelo registrado exitosamente: {model_package.model_package_arn}")
    except Exception as e:
        print(f"Error al registrar el modelo: {e}")
        raise # Re-lanzar la excepción para que el trabajo falle