# convert_dataset.py
import json
import os
import librosa
import argparse
from tqdm import tqdm

def convert_metadata(original_path, new_path):
    print(f"Leyendo metadata original de: {original_path}")
    with open(original_path, 'r', encoding='utf-8') as f:
        original_lines = f.readlines()

    with open(new_path, 'w', encoding='utf-8') as f_out:
        print(f"Procesando {len(original_lines)} archivos para el nuevo formato...")
        for line in tqdm(original_lines):
            data = json.loads(line)
            audio_filename = data['file_name']
            # Asumimos que los audios están en la misma carpeta que el metadata original
            audio_filepath = os.path.join(os.path.dirname(original_path), audio_filename)

            if os.path.exists(audio_filepath):
                duration = librosa.get_duration(path=audio_filepath)
                new_entry = {
                    "audio_filepath": audio_filepath,
                    "description": data['text'],
                    "duration": duration
                }
                f_out.write(json.dumps(new_entry) + '\n')
            else:
                print(f"AVISO: No se encontró el archivo {audio_filepath}, se omitirá.")

    print(f"¡Conversión completada! Nuevo archivo de metadatos guardado en: {new_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convierte metadata.jsonl al formato de Audiocraft.")
    parser.add_argument("original_file", type=str, help="Ruta al archivo metadata.jsonl original.")
    parser.add_argument("new_file", type=str, help="Ruta donde se guardará el nuevo archivo train.jsonl.")
    args = parser.parse_args()

    convert_metadata(args.original_file, args.new_file)
