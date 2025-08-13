import pandas as pd
from sklearn.datasets import load_iris
import os

# Cargar datos
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Ruta de salida estándar en SageMaker
output_dir = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/processing/output')
df.to_csv(os.path.join(output_dir, 'iris_raw.csv'), index=False)
print("Datos guardados correctamente.")
