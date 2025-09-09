# prepare_dataset.py
import os
import json
import argparse
import logging
from typing import List, Dict, Any

import torch
# Asegúrate de tener estas librerías: pip install librosa transformers
try:
    import librosa
    from transformers import AutoProcessor, AutoModelForAudioClassification
except ImportError as e:
    logging.warning(f"Dependencias para audio-tagging no encontradas. Instálalas si es necesario. Error: {e}")

logger = logging.getLogger(__name__)

class AudioTagger:
    """Componente para etiquetar archivos de audio y generar metadata para MusicGen."""
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        self.processor = None
        self.model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Usando dispositivo: {self.device}")

    def load_model(self):
        """Cargar el modelo de audio tagging."""
        try:
            print(f"Cargando modelo de tagging '{self.model_name}'...")
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print("Modelo de tagging cargado correctamente.")
            return True
        except Exception as e:
            print(f"Error cargando modelo de tagging: {str(e)}")
            return False

    def generate_caption_from_file(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Procesa un único archivo de audio y genera un caption."""
        if not self.model or not self.processor:
            return {"error": "El modelo de tagging no está cargado."}
        try:
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            
            labels = []
            for prob, idx in zip(top_probs, top_indices):
                label = self.model.config.id2label[idx.item()]
                labels.append(label)
            
            caption = ", ".join(labels)
            return {
                "file_name": os.path.basename(audio_path),
                "text": caption
            }
        except Exception as e:
            return {"error": f"Error procesando '{audio_path}': {str(e)}"}

def process_directory(dataset_path: str):
    """Escanea un directorio, procesa los audios y crea metadata.jsonl."""
    tagger = AudioTagger()
    if not tagger.load_model():
        return

    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = [f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in supported_extensions]

    if not audio_files:
        print(f"No se encontraron archivos de audio en '{dataset_path}'")
        return

    output_file = os.path.join(dataset_path, "metadata.jsonl")
    
    print(f"Se encontraron {len(audio_files)} archivos de audio. Procesando...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, audio_file in enumerate(sorted(audio_files)): # sorted() para un orden predecible
            full_path = os.path.join(dataset_path, audio_file)
            print(f"[{i+1}/{len(audio_files)}] Procesando: {audio_file}")
            
            result = tagger.generate_caption_from_file(full_path)
            
            if "error" not in result:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    print(f"\n¡Proceso completado! Archivo de metadatos guardado en: {output_file}")
    print("Puedes editar este archivo manualmente para mejorar las descripciones si lo deseas.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Genera automáticamente 'metadata.jsonl' para un dataset de audio.")
    parser.add_argument("dataset_path", type=str, help="Ruta a la carpeta que contiene tus archivos de audio.")
    args = parser.parse_args()
    
    if not os.path.isdir(args.dataset_path):
        print(f"Error: La ruta '{args.dataset_path}' no es un directorio válido.")
    else:
        process_directory(args.dataset_path)