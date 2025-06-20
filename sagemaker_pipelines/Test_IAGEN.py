import json
import openai
import os


def cargar_credenciales(ruta_credenciales):
    """Carga el archivo de credenciales JSON para Azure OpenAI."""
    try:
        with open(ruta_credenciales, "r") as file:
            creds = json.load(file)
        return creds
    except Exception as e:
        raise RuntimeError(f"Error al cargar credenciales: {e}")


def configurar_openai_azure(creds):
    """Configura las variables necesarias para conectarse a Azure OpenAI."""
    openai.api_type = "azure"
    openai.api_version = creds.get("AZURE_API_VERSION")
    openai.api_key = creds.get("AZURE_API_KEY")
    openai.api_base = creds.get("AZURE_ENDPOINT")  


def generar_descripcion_con_azure(prompt, engine="gpt-35-turbo-16k-PQR", temperatura=0.3, max_tokens=150):
    """Genera una respuesta desde Azure OpenAI."""
    try:
        response = openai.ChatCompletion.create(
            engine=engine,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperatura,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error al generar descripción: {e}"


def main():
    ruta_credenciales = "/Users/dbarr15/Documents/IAGEN/credentials.json"  # Ajusta si estás en otro entorno
    prompt = "Eres un ingeniero experto en mlops. ¿Resume cuáles son los etapas clave del ciclo de vida de MLOps?"

    creds = cargar_credenciales(ruta_credenciales)
    configurar_openai_azure(creds)
    respuesta = generar_descripcion_con_azure(prompt)
    print("\nRespuesta del modelo:\n", respuesta)


if __name__ == "__main__":
    main()