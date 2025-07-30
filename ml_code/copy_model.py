import shutil
import os

# Copia directa porque ya está en /opt/ml/processing/input como input del paso
input_path = "/opt/ml/processing/input/model.tar.gz"
output_path = "/opt/ml/processing/input/model.tar.gz"  # Se queda en el mismo lugar, el pipeline lo copia a S3

# Verificación opcional
if os.path.exists(input_path):
    print(f"Archivo encontrado y será copiado: {input_path}")
else:
    raise FileNotFoundError("El archivo model.tar.gz no se encontró en /opt/ml/processing/input/")
