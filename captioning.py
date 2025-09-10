import os
import json
import logging
from typing import List, Dict, Any

import torch
try:
    import librosa
    from transformers import AutoProcessor, AutoModelForAudioClassification
except ImportError as e:
    logging.warning(f"Some audio tagging dependencies missing: {e}")

logger = logging.getLogger(__name__)

class AudioTagger:
    """Componente para etiquetar archivos de audio"""
    def __init__(self):
        self.processor = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        """Cargar modelo de audio tagging"""
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(model_name)
            self.model.to(self.device)
            self.model.eval()
            return "Modelo de audio tagging cargado correctamente"
        except Exception as e:
            return f"Error cargando modelo: {str(e)}"

    def process_audio_file(self, audio_path: str, top_k: int = 5) -> Dict[str, Any]:
        """Procesar un archivo de audio y generar etiquetas"""
        try:
            if not self.model or not self.processor:
                return {"error": "Modelo no cargado"}
            # Cargar audio
            audio, sr = librosa.load(audio_path, sr=16000)
            # Procesar con el modelo
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            # Obtener top-k etiquetas
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            labels = []
            for i, (prob, idx) in enumerate(zip(top_probs, top_indices)):
                label = self.model.config.id2label[idx.item()]
                confidence = prob.item()
                labels.append({"label": label, "confidence": confidence})
            # Generar caption
            caption = self.generate_caption(labels)
            return {
                "caption": caption,
                "labels": labels,
                "file_path": audio_path
            }
        except Exception as e:
            return {"error": f"Error procesando audio: {str(e)}"}

    def generate_caption(self, labels: List[Dict]) -> str:
        """Generar caption basado en las etiquetas"""
        if not labels:
            return "Audio sin clasificar"
        # Tomar las 3 etiquetas principales
        main_labels = [label["label"] for label in labels[:3]]
        caption = f"Audio containing {', '.join(main_labels)}"
        return caption

    def process_batch(self, audio_files: List[str], output_dir: str, top_k: int = 5) -> str:
        """Procesar múltiples archivos de audio"""
        results = []
        for audio_file in audio_files:
            # CORRECCIÓN: Pasar la ruta completa del archivo
            result = self.process_audio_file(audio_file, top_k)
            if "error" not in result:
                results.append({
                    "file": os.path.basename(audio_file),
                    "caption": result["caption"],
                    "labels": result["labels"]
                })
        # Guardar resultados
        output_file = os.path.join(output_dir, "captions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        return f"Procesados {len(results)} archivos. Resultados guardados en {output_file}"