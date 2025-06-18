import pandas as pd
from sklearn.datasets import load_iris
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    args = parser.parse_args()

    # Cargar datos Iris (ejemplo simplificado: directamente de scikit-learn)
    data = load_iris()
    X, y = data.data, data.target
    feature_names = data.feature_names
    # target_name = data.target_names[0] # Solo para referencia, no se usa directamente

    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y

    # Guardar los datos crudos en el directorio de salida de SageMaker
    raw_data_path = os.path.join(args.output_data_dir, 'iris_raw.csv')
    df.to_csv(raw_data_path, index=False)
    print(f"Raw data saved to {raw_data_path}")