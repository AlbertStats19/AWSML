import shutil
import os

# Rutas en el entorno de SageMaker Processing
input_path = "/opt/ml/processing/model_input/model.tar.gz"
output_path = "/opt/ml/processing/model_output/model.tar.gz"

# Verifica si el modelo existe y lo copia al directorio de salida
if os.path.exists(input_path):
    shutil.copy(input_path, output_path)
    print(f"✅ Modelo copiado correctamente a: {output_path}")
else:
    raise FileNotFoundError("❌ El archivo model.tar.gz no se encontró en /opt/ml/processing/input/")
