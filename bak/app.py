# -*- coding: utf-8 -*-
# app.py (Versión 4.6 – Caption completo + Flash‑Attention + arquitectura estable)

"""
Flujo completo
---------------
1️⃣  Gestor de prompts (guardar / cargar / eliminar).  
2️⃣  Mejora de prompts con Ollama (opcionalmente con palabras clave del
   `metadata.jsonl`).  
3️⃣  Generación de `metadata.jsonl` usando el AudioTagger de `captioning.py`.  
   Cada línea tiene el siguiente formato **exacto**:

   {
       "key": "",
       "artist": "Voyager I",
       "sample_rate": 44100,
       "file_extension": "mp3",
       "description": "<caption generada>",
       "keywords": "",
       "duration": 20.0,
       "bpm": "",
       "genre": "electronic",
       "title": "Untitled song",
       "name": "electro_2",
       "instrument": "Mix",
       "moods": []
   }

4️⃣  Entrenamiento DreamBooth (el script ahora usa `--text_column_name=description`).  
5️⃣  Inferencia: selección dinámica del LoRA + generación de audio con MusicGen
   (Flash‑Attention activado, **excepto** el encoder del T5).  
"""

import json
import logging
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple, Optional
import numpy as np
import gradio as gr
import librosa
import requests
import torch
import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
import soundfile as sf

# --------------------------------------------------------------------------- #
# IMPORT DE AUDIO‑TAGGER ------------------------------------------------------ #
# --------------------------------------------------------------------------- #
# El tagger está definido en `captioning.py`.  Sólo lo importamos aquí.
from captioning import AudioTagger

# --------------------------------------------------------------------------- #
# LOGGING ------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN GLOBAL ------------------------------------------------------ #
# --------------------------------------------------------------------------- #
SETTINGS_FILE = "settings.json"
MODEL_ID = "facebook/musicgen-small"

# --------------------------------------------------------------------------- #
# UTILIDADES --------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def load_settings() -> Dict[str, Any]:
    """Carga `settings.json` o devuelve valores por defecto."""
    defaults = {
        "dataset_path": "",
        "output_dir": "./mi_lora_final",
        "epochs": 15,
        "lr": 0.0001,
        "lora_r": 32,
        "lora_alpha": 64,
        "max_duration": 180,
        "ollama_model": "",
        "train_seed": 42,
        "inference_prompt": "80s rock ballad with a power guitar solo",
        "inference_duration": 15,
        "lora_path": "",
        "inference_seed": -1,
        "guidance_scale": 3.0,
        "temperature": 1.0,
        "top_k": 250,
        "top_p": 0.0,
    }
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                overrides = json.load(f)
                defaults.update(overrides)
        except (json.JSONDecodeError, IOError):
            logger.warning(
                "No se pudo leer settings.json – se usan valores por defecto"
            )
    return defaults


def save_settings(settings: Dict[str, Any]) -> None:
    """Persiste `settings.json` en disco."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        logger.info("✅ Settings guardados")
    except IOError as e:
        logger.error(f"Error guardando settings: {e}")


settings = load_settings()

# --------------------------------------------------------------------------- #
# CARGA DEL MODELO BASE + PROCESSOR ---------------------------------------- #
# --------------------------------------------------------------------------- #
print("Cargando procesador y modelo base (CPU)…")
processor = AutoProcessor.from_pretrained(MODEL_ID)

base_model = MusicgenForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,               # ← argumento correcto (torch_dtype está deprecado)
)

# --------------------------------------------------------------------------- #
# FLASH‑ATTENTION ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Activamos Flash‑Attention 2 en todo el modelo y la desactivamos sólo en el
# encoder de texto (T5) para evitar incompatibilidades.
if hasattr(base_model.config, "use_flash_attention_2"):
    base_model.config.use_flash_attention_2 = True
    logger.info("⚡ Flash‑Attention 2 activado en MusicGen.")
if hasattr(base_model, "text_encoder"):
    if hasattr(base_model.text_encoder.config, "use_flash_attention_2"):
        base_model.text_encoder.config.use_flash_attention_2 = False
        logger.info(
            "⚡ Flash‑Attention 2 **desactivado** en el encoder del texto (T5)."
        )
base_model.to("cpu")
print("✔️ Modelo base listo")

# --------------------------------------------------------------------------- #
# CLASES DE LÓGICA (copiadas sin cambios de la v4.0) ------------------------ #
# --------------------------------------------------------------------------- #
class PromptManager:
    """Gestor para guardar, cargar y eliminar prompts."""
    def __init__(self, prompts_file="saved_prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()

    def load_prompts(self):
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError):
                logger.error("Error al leer saved_prompts.json")
        return []

    def save_prompts(self):
        try:
            with open(self.prompts_file, "w", encoding="utf-8") as f:
                json.dump(self.prompts, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error guardando prompts: {e}")

    def update_prompt(self, name, text):
        found = False
        for p in self.prompts:
            if p["name"] == name:
                p["text"] = text
                found = True
                break
        if not found and name.strip() and text.strip():
            self.prompts.append(
                {"name": name, "text": text, "created": datetime.now().isoformat()}
            )
        self.save_prompts()
        return f"Prompt '{name}' guardado/actualizado."

    def delete_prompt(self, name):
        before = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["name"] != name]
        if len(self.prompts) < before:
            self.save_prompts()
            return f"Prompt '{name}' eliminado."
        return f"Prompt '{name}' no encontrado."

    def get_prompt_text(self, name):
        for p in self.prompts:
            if p["name"] == name:
                return p["text"]
        return ""

    def get_prompt_names(self):
        return sorted([p["name"] for p in self.prompts])


class OllamaIntegration:
    """Integración con Ollama para mejorar y traducir prompts."""
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def get_available_models(self):
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
            return []
        except requests.ConnectionError:
            return ["Error: No se pudo conectar a Ollama"]
        except Exception as e:
            logger.error(f"Error inesperado en Ollama: {e}")
            return []

    def unload_ollama_model(self, model_name: str):
        if not model_name or "Error" in model_name:
            return "Selecciona un modelo válido para descargar."
        try:
            subprocess.run(
                ["ollama", "stop", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            return f"✅ Modelo '{model_name}' descargado de la memoria."
        except Exception:
            return f"⚠️ No se pudo descargar '{model_name}' (quizá no estaba cargado)."

    def enhance_and_translate_prompt(
        self,
        model_name: str,
        base_prompt: str,
        use_captions_context: bool,
        dataset_path: str,
    ) -> str:
        if not base_prompt.strip():
            return "El prompt base no puede estar vacío."

        # --------------------------------------------------------------- #
        # 1️⃣ Construir contexto a partir de `metadata.jsonl` (si se pide)
        # --------------------------------------------------------------- #
        tags_context = ""
        if use_captions_context and dataset_path and os.path.isdir(dataset_path):
            metadata_file = os.path.join(dataset_path, "metadata.jsonl")
            if os.path.exists(metadata_file):
                try:
                    kw_set = set()
                    with open(metadata_file, "r", encoding="utf-8") as f:
                        for line in f:
                            data = json.loads(line)
                            # usamos el campo `description` porque ese es el que
                            # contiene la caption real
                            kw_set.update(
                                re.findall(r"\b\w+\b", data.get("description", "").lower())
                            )
                    if kw_set:
                        tags_context = (
                            "Inspírate en estas palabras clave del dataset: "
                            + ", ".join(sorted(list(kw_set)))
                            + ". "
                        )
                except Exception as e:
                    logger.error(f"Error leyendo metadata.jsonl: {e}")

        # --------------------------------------------------------------- #
        # 2️⃣ Prompt para Ollama
        # --------------------------------------------------------------- #
        ollama_prompt = (
            f"You are an expert music‑prompt writer. {tags_context}"
            f"Improve and translate this idea into a concise English prompt for MusicGen: \"{base_prompt}\""
        )
        payload = {"model": model_name, "prompt": ollama_prompt, "stream": False}
        try:
            r = self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=60
            )
            r.raise_for_status()
            data = r.json()
            out = data.get("response", "").strip().replace('"', "")
            return out[0].upper() + out[1:] if out else base_model
        except Exception as e:
            logger.error(f"Error en Ollama: {e}")
            return "Error al conectar con Ollama."


# --------------------------------------------------------------------------- #
# INSTANCIACIÓN GLOBAL ------------------------------------------------------ #
# --------------------------------------------------------------------------- #
prompt_manager = PromptManager()
ollama = OllamaIntegration()
available_ollama_models = ollama.get_available_models()
if not settings["ollama_model"] and available_ollama_models:
    settings["ollama_model"] = available_ollama_models[0]

# --------------------------------------------------------------------------- #
# AUDIO TAGGER – Carga única ------------------------------------------------ #
# --------------------------------------------------------------------------- #
tagger = AudioTagger()
tagger_loaded = False
try:
    load_msg = tagger.load_model()
    logger.info(load_msg)
    if "Error" not in load_msg:
        tagger_loaded = True
except Exception as e:
    logger.error(f"Error al iniciar AudioTagger: {e}")

# --------------------------------------------------------------------------- #
# AYUDA PARA EXTRAER LA CAPTION --------------------------------------------- #
# --------------------------------------------------------------------------- #
def _extract_description(raw: Any, audio_path: str = None) -> dict:
    """
    Extrae información completa del tagger y audio para crear un caption rico.
    """
    result = {
        "description": "audio caption placeholder",
        "keywords": [],
        "bpm": "",
        "genre": "electronic",
        "moods": [],
        "instruments": ["Mix"],
        "timbre": "",
        "dynamics": "",
        "structure": ""
    }
    
    # Extraer descripción del tagger
    description_text = ""
    if isinstance(raw, dict):
        for key in ("caption", "text", "label", "labels", "tags", "predictions", "prediction", "description"):
            if key in raw and raw[key]:
                val = raw[key]
                if isinstance(val, (list, tuple)):
                    description_text = ", ".join([str(v).strip() for v in val if v])
                else:
                    description_text = str(val).strip()
                break
    elif isinstance(raw, (list, tuple)):
        description_text = ", ".join([str(v).strip() for v in raw if v])
    elif isinstance(raw, str):
        description_text = raw.strip()
    
    result["description"] = description_text if description_text else "audio caption placeholder"
    
    # Si tenemos la ruta del archivo, podemos hacer análisis adicional
    if audio_path and os.path.exists(audio_path):
        try:
            # Análisis con librosa
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            duration = librosa.get_duration(y=y, sr=sr)
            
            # BPM - manejar correctamente los arrays de numpy
            tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
            # tempo puede ser un array, tomamos el primer valor o el promedio
            if isinstance(tempo, np.ndarray):
                tempo_val = float(tempo[0]) if len(tempo) > 0 else 0.0
            else:
                tempo_val = float(tempo) if tempo > 0 else 0.0
            result["bpm"] = str(round(tempo_val)) if tempo_val > 0 else ""
            
            # Spectral features para timbre
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            avg_centroid = float(np.mean(spectral_centroids))
            if avg_centroid < 2000:
                timbre = "dark"
            elif avg_centroid < 4000:
                timbre = "warm"
            else:
                timbre = "bright"
            result["timbre"] = timbre
            
            # RMS para dinámica
            rms = librosa.feature.rms(y=y)[0]
            avg_rms = float(np.mean(rms))
            if avg_rms < 0.05:
                dynamics = "soft"
            elif avg_rms < 0.15:
                dynamics = "medium"
            else:
                dynamics = "loud"
            result["dynamics"] = dynamics
            
            # Estructura básica por duración
            if duration < 30:
                structure = "loop"
            elif duration < 120:
                structure = "intro"
            else:
                structure = "full track"
            result["structure"] = structure
            
            # Keywords basadas en análisis
            keywords = []
            if tempo_val > 120:
                keywords.append("energetic")
            if "bright" in timbre:
                keywords.append("crisp")
            if "loud" in dynamics:
                keywords.append("powerful")
            if duration > 300:  # 5 minutos
                keywords.append("epic")
            result["keywords"] = keywords
            
            # Mood básico
            if tempo_val > 100:
                moods = ["uplifting", "energetic"]
            elif tempo_val > 60:
                moods = ["calm", "reflective"]
            else:
                moods = ["ambient", "meditative"]
            result["moods"] = moods
            
        except Exception as e:
            logger.warning(f"Error en análisis de audio {audio_path}: {e}")
            logger.debug(f"Tipo de error: {type(e).__name__}")
    
    return result


# --------------------------------------------------------------------------- #
# FUNCIÓN DE METADATA ------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def generate_metadata(dataset_dir: str) -> str:
    """
    Recorre `dataset_dir` y crea un `metadata.jsonl` con información completa.
    """
    if not tagger_loaded:
        return "❌ El modelo de audio-tagging no está disponible. Revisa los logs."

    root = Path(dataset_dir)
    if not root.is_dir():
        return f"❌ Ruta de dataset inválida: {dataset_dir}"

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    if not audio_files:
        return f"⚠️ No se encontraron archivos de audio en {dataset_dir}"

    out_path = root / "metadata.jsonl"
    success_count = 0
    error_count = 0
    
    with out_path.open("w", encoding="utf-8") as out_f:
        for audio_path in tqdm.tqdm(
            audio_files, desc="Generando metadata con captioning.py"
        ):
            try:
                raw_res = tagger.process_audio_file(str(audio_path))
            except AttributeError:
                try:
                    raw_res = tagger(str(audio_path))
                except Exception:
                    raw_res = getattr(
                        tagger, "generate_caption_from_file", lambda x: {}
                    )(str(audio_path))

            logger.debug(
                f"Salida raw del tagger para {audio_path.name}: {raw_res}"
            )

            try:
                y, sr = librosa.load(str(audio_path), sr=None, mono=False)
                duration = float(librosa.get_duration(y=y, sr=sr))
            except Exception as e:
                logger.warning(f"Librosa falló para {audio_path.name}: {e}")
                sr = 44100
                duration = 0.0

            if isinstance(raw_res, dict) and "error" in raw_res:
                logger.warning(
                    f"Error del tagger en {audio_path.name}: {raw_res['error']}"
                )
                audio_info = _extract_description({}, str(audio_path))
                error_count += 1
            else:
                audio_info = _extract_description(raw_res, str(audio_path))
                success_count += 1

            # Construir caption completo
            description_parts = []
            if audio_info["description"] and audio_info["description"] != "audio caption placeholder":
                description_parts.append(audio_info["description"])
            if audio_info["timbre"]:
                description_parts.append(f"{audio_info['timbre']} timbre")
            if audio_info["dynamics"]:
                description_parts.append(f"{audio_info['dynamics']} dynamics")
            if audio_info["structure"]:
                description_parts.append(audio_info["structure"])
            
            full_description = ", ".join(description_parts) if description_parts else "audio caption placeholder"

            caption_dict = {
                "key": "",
                "artist": "Tonetxo",
                "sample_rate": int(sr),
                "file_extension": audio_path.suffix.lstrip("."),
                "description": full_description,
                "keywords": ", ".join(audio_info["keywords"]) if audio_info["keywords"] else "",
                "duration": round(float(duration), 2),
                "bpm": audio_info["bpm"],
                "genre": audio_info["genre"],
                "title": audio_path.stem.replace("_", " ").title(),
                "name": audio_path.stem,
                "instrument": ", ".join(audio_info["instruments"]) if audio_info["instruments"] else "Mix",
                "moods": audio_info["moods"],
                "file_name": audio_path.name,
                "audio_filepath": str(audio_path)
            }
            
            # Agregar campos adicionales si existen
            if audio_info.get("timbre"):
                caption_dict["timbre"] = audio_info["timbre"]
            if audio_info.get("dynamics"):
                caption_dict["dynamics"] = audio_info["dynamics"]
            if audio_info.get("structure"):
                caption_dict["structure"] = audio_info["structure"]

            out_f.write(json.dumps(caption_dict, ensure_ascii=False) + "\n")

    return f"✅ metadata.jsonl creado en: {out_path} (Éxito: {success_count}, Errores: {error_count})"

# === NUEVA FUNCIÓN: Augmentar Dataset ===
# === CORREGIDO: Augmentar Dataset ===
def augment_dataset_simple(input_dataset_path: str, output_dataset_path: str) -> str:
    """
    Augmenta un dataset de audio aplicando transformaciones simples.
    """
    logger.info(f"Iniciando augmentación de dataset: {input_dataset_path} -> {output_dataset_path}")
    
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)
    augmented_audio_dir = output_path / "augmented_audio"
    
    # Validar directorio de entrada
    if not input_path.is_dir():
        return f"❌ Directorio de entrada inválido: {input_dataset_path}"

    metadata_file = input_path / "metadata.jsonl"
    if not metadata_file.exists():
        return f"❌ No se encontró 'metadata.jsonl' en {input_dataset_path}"

    # Crear directorios de salida
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        augmented_audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        return f"❌ Error creando directorio de salida: {e}"

    def simple_augment_audio(y, sr, filename_prefix):
        """Aplica augmentaciones simples al audio."""
        augmented_samples = []
        
        # Pitch Shift
        try:
            y_pitch_down = librosa.effects.pitch_shift(y, sr=sr, n_steps=-1)
            augmented_samples.append((y_pitch_down, sr, f"{filename_prefix}_pitch_down1"))
        except Exception:
            pass # Omitir si falla
            
        try:
            y_pitch_up = librosa.effects.pitch_shift(y, sr=sr, n_steps=1)
            augmented_samples.append((y_pitch_up, sr, f"{filename_prefix}_pitch_up1"))
        except Exception:
            pass
            
        # Time Stretch
        try:
            y_stretch_slow = librosa.effects.time_stretch(y, rate=0.98)
            augmented_samples.append((y_stretch_slow, sr, f"{filename_prefix}_stretch_slow"))
        except Exception:
            pass
            
        try:
            y_stretch_fast = librosa.effects.time_stretch(y, rate=1.02)
            augmented_samples.append((y_stretch_fast, sr, f"{filename_prefix}_stretch_fast"))
        except Exception:
            pass
            
        # Volumen
        y_vol_down = np.clip(y * 0.9, -1.0, 1.0)
        augmented_samples.append((y_vol_down, sr, f"{filename_prefix}_vol_down"))
        
        y_vol_up = np.clip(y * 1.1, -1.0, 1.0)
        augmented_samples.append((y_vol_up, sr, f"{filename_prefix}_vol_up"))
        
        return augmented_samples

    total_samples = 0
    new_metadata_path = output_path / "metadata.jsonl"
    
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f_in, \
             open(new_metadata_path, 'w', encoding='utf-8') as f_out:
            
            # Leer todas las líneas originales
            original_lines = [line.strip() for line in f_in if line.strip()]
            
            # 1. Copiar muestras originales PERO con rutas actualizadas
            logger.info(f"Procesando {len(original_lines)} muestras originales...")
            for line in original_lines:
                try:
                    data = json.loads(line)
                                                            # --- INICIO DEL BLOQUE CORREGIDO PARA MUESTRAS ORIGINALES ---
                    # 1. Definir la ruta absoluta del archivo original
                    original_audio_path_str = data['audio_filepath']
                    original_audio_path = Path(original_audio_path_str)
                    
                    # Si la ruta en el JSON original es relativa, convertirla a absoluta
                    # basándonos en el directorio del dataset original
                    if not original_audio_path.is_absolute():
                        original_audio_path = input_path / original_audio_path
                        
                    if not original_audio_path.exists():
                        logger.warning(f"Archivo original no encontrado, saltando: {original_audio_path}")
                        continue
                        
                    # 2. Definir la ruta donde se copiará el archivo (y su ruta absoluta)
                    original_filename = data['file_name']
                    new_audio_path_obj = augmented_audio_dir / original_filename
                    new_audio_path_absolute_str = str(new_audio_path_obj.resolve()) # <-- .resolve() para ruta absoluta canónica
                    
                    # 3. Copiar el archivo
                    import shutil
                    shutil.copy2(original_audio_path, new_audio_path_obj) # <-- Copiar a la ruta Path
                    logger.debug(f"Copiado: {original_audio_path} -> {new_audio_path_obj}")
                    
                    # 4. Crear una nueva entrada para el metadata.jsonl copiado
                    new_data_entry = data.copy()
                    # *** LÍNEA CRÍTICA CORREGIDA ***
                    new_data_entry['audio_filepath'] = new_audio_path_absolute_str # <-- RUTA ABSOLUTA
                    
                    # 5. Escribir la entrada actualizada en el nuevo metadata.jsonl
                    f_out.write(json.dumps(new_data_entry, ensure_ascii=False) + '\n')
                    total_samples += 1
                    # --- FIN DEL BLOQUE CORREGIDO PARA MUESTRAS ORIGINALES ---
                        
                except Exception as e:
                    logger.error(f"Error procesando muestra original: {e}")
                    continue
            
            logger.info(f"Copiadas {len(original_lines)} muestras originales")
            
            # 2. Augmentar cada muestra original
            for i, line in enumerate(tqdm.tqdm(original_lines, desc="Augmentando")):
                try:
                    data = json.loads(line)
                    original_audio_path = Path(data['audio_filepath'])
                    
                    # Verificar que el archivo existe
                    if not original_audio_path.exists():
                        logger.warning(f"Archivo no encontrado, saltando: {original_audio_path}")
                        continue
                        
                    # Cargar audio
                    y, sr = librosa.load(str(original_audio_path), sr=None, mono=True)
                    
                    # Generar augmentaciones
                    filename_prefix = original_audio_path.stem
                    augmented_samples = simple_augment_audio(y, sr, filename_prefix)
                    
                    # Guardar cada augmentación
                    for aug_y, aug_sr, aug_name in augmented_samples:
                        try:
                                                        # --- INICIO DEL BLOQUE CORREGIDO PARA MUESTRAS AUGMENTADAS ---
                            # 1. Generar el nombre de archivo y la ruta Path
                            new_audio_filename = f"{aug_name}.wav"
                            new_audio_path_obj = augmented_audio_dir / new_audio_filename
                            new_audio_path_absolute_str = str(new_audio_path_obj.resolve()) # <-- .resolve() para ruta absoluta canónica
                            
                            # 2. Guardar el audio augmentado
                            sf.write(new_audio_path_absolute_str, aug_y, aug_sr) # <-- Usar la ruta absoluta para guardar
                            logger.debug(f"Guardado augmentado: {new_audio_path_absolute_str}")
                            
                            # 3. Crear nueva entrada en metadata con RUTA ABSOLUTA
                            new_data = data.copy()
                            new_data['name'] = aug_name
                            new_data['title'] = f"{data['title']} ({aug_name.split('_')[-1]})"
                            new_data['file_name'] = new_audio_filename
                            # *** LÍNEA CRÍTICA CORREGIDA ***
                            new_data['audio_filepath'] = new_audio_path_absolute_str # <-- Usar la ruta absoluta
                            
                            new_data['description'] = f"{data['description']} (augmented)"
                            
                            # 4. Escribir la entrada en el nuevo metadata.jsonl
                            f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                            total_samples += 1
                            # --- FIN DEL BLOQUE CORREGIDO PARA MUESTRAS AUGMENTADAS ---                            
                            
                        except Exception as e:
                            logger.error(f"Error guardando augmentación {aug_name}: {e}")
                            continue
                            
                except Exception as e:
                    logger.error(f"Error procesando muestra {i}: {e}")
                    continue
    
        logger.info(f"✅ Dataset augmentado: {total_samples} muestras totales")
        return f"✅ Dataset augmentado: {total_samples} muestras generadas en {output_path}"
        
    except Exception as e:
        logger.error(f"Error en augmentación: {e}")
        return f"❌ Error en augmentación: {str(e)}"

# === NUEVA FUNCIÓN: Guardar audio generado ===
def save_generated_audio(audio_data: Tuple[int, Any], output_dir: str = "./generated_audio") -> str:
    """
    Guarda el audio generado en formato WAV.
    
    Args:
        audio_data: Tupla (sample_rate, audio_numpy_array) del componente gr.Audio
        output_dir: Directorio donde guardar el audio
    
    Returns:
        str: Ruta del archivo guardado o mensaje de error
    """
    if audio_data is None:
        return "❌ No hay audio para guardar"
    
    sr, audio_np = audio_data
    
    if audio_np is None or len(audio_np) == 0:
        return "❌ Audio vacío"
    
    # Crear directorio si no existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Generar nombre de archivo único
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"musicgen_generated_{timestamp}.wav"
    filepath = os.path.join(output_dir, filename)
    
    try:
        # soundfile espera (time, channels) para estéreo
        if audio_np.ndim == 2 and audio_np.shape[1] <= 2:
            # Ya está en formato correcto (time, channels)
            pass
        elif audio_np.ndim == 2 and audio_np.shape[0] <= 2:
            # Está en (channels, time), hay que transponer
            audio_np = audio_np.T
        # Para mono, debe ser 1D
        elif audio_np.ndim == 1:
            pass # Correcto
        else:
            # Fallback: intentar squeeze
            audio_np = audio_np.squeeze()
            
        sf.write(filepath, audio_np, sr)
        logger.info(f"✅ Audio guardado en: {filepath}")
        return f"✅ Guardado: {filepath}"
    except Exception as e:
        error_msg = f"❌ Error guardando audio: {str(e)}"
        logger.error(error_msg)
        return error_msg
# --------------------------------------------------------------------------- #
# ENTRENAMIENTO (DreamBooth) ------------------------------------------------ #
# --------------------------------------------------------------------------- #
# === CORREGIDO: ENTRENAMIENTO (DreamBooth) ===
import time # Asegúrate de que 'time' esté importado

def modify_and_run_training(
    dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, train_seed,
    use_augmented=False, augmented_path=""
) -> Generator[str, None, None]:
    """
    Modifica y ejecuta el script de entrenamiento, mostrando logs acumulados en tiempo real.
    """
    # Si se debe usar el dataset augmentado:
        # === CORREGIDO: Determinar y definir siempre absolute_dataset_path ===
    # Elegir la ruta del dataset a usar
    if use_augmented and augmented_path and os.path.exists(augmented_path):
        dataset_path_to_use = augmented_path
        logger.info(f"Usando dataset augmentado: {dataset_path_to_use}")
    else:
        dataset_path_to_use = dataset_path
        # Opcional: logger.info(f"Usando dataset original: {dataset_path_to_use}")

    # Convertir la ruta elegida a una ruta absoluta
    # Esta línea SIEMPRE se ejecuta, definiendo absolute_dataset_path
    absolute_dataset_path = os.path.abspath(dataset_path_to_use)
    logger.info(f"Usando ruta de dataset absoluta: {absolute_dataset_path}")
    # === FIN DE LA CORRECCIÓN ===
    script_path = "./musicgen-dreamboothing/dreambooth_musicgen.py"
    try:
        with open(script_path, "r", encoding="utf-8") as f:
            script_content = f.read()
        script_content = re.sub(r"r=\d+", f"r={int(lora_r)}", script_content)
        script_content = re.sub(r"lora_alpha=\d+", f"lora_alpha={int(lora_alpha)}", script_content)
        script_content = re.sub(r"--text_column_name=\w+", "--text_column_name=description", script_content)
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        yield f"✅ Script modificado: r={int(lora_r)}, lora_alpha={int(lora_alpha)}\n"
    except Exception as e:
        yield f"❌ Error al modificar script: {e}\n"
        return

    command = [
        "accelerate", "launch", "dreambooth_musicgen.py",
        f"--model_name_or_path={MODEL_ID}", f"--dataset_name={absolute_dataset_path}",
        f"--output_dir={output_dir}", f"--num_train_epochs={int(epochs)}",
        "--use_lora", f"--learning_rate={lr}", "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=4", "--fp16", "--text_column_name=description",
        "--target_audio_column_name=audio_filepath", "--train_split_name=train",
        "--overwrite_output_dir", "--do_train", "--decoder_start_token_id=2048",
        f"--max_duration_in_seconds={int(max_duration)}", "--gradient_checkpointing",
        f"--seed={int(train_seed)}",
        "--logging_steps=1",
        "--logging_strategy=steps", 
        "--logging_first_step=True",
    ]

    # Inicializar el acumulador de logs
    full_log = "🚀 Lanzando entrenamiento...\n\n"
    yield full_log

    process = subprocess.Popen(
        command,
        cwd="./musicgen-dreamboothing",
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        bufsize=1,
        universal_newlines=True
    )

    # Leer la salida línea por línea en tiempo real
    try:
        for line in iter(process.stdout.readline, ''):
            if line:  # Solo procesar líneas no vacías
                # Limpiar la línea de retornos de carro al final
                clean_line = line.rstrip('\n\r')
                if clean_line: # Solo añadir si después de limpiar no está vacía
                    # Añadir la nueva línea al log acumulado
                    full_log += clean_line + "\n"
                    # Yield el log acumulado completo para actualizar la UI
                    yield full_log
        # El iterador termina cuando el proceso cierra stdout
    except Exception as e:
        error_line = f"[ERROR - Excepción al leer stdout] {e}\n"
        full_log += error_line
        yield full_log
    finally:
        # Esperar a que el proceso termine
        process.wait()

    # Añadir el resultado final
    if process.returncode == 0:
        final_msg = "\n✅ ¡Entrenamiento finalizado exitosamente!"
    else:
        final_msg = f"\n❌ Proceso terminó con código de error: {process.returncode}"

    full_log += final_msg
    yield full_log






# --------------------------------------------------------------------------- #
# INFERENCIA – Gestión de LoRA ---------------------------------------------- #
# --------------------------------------------------------------------------- #
def switch_model_and_state(lora_files: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Carga un LoRA a partir de una lista de archivos temporales de Gradio.
    """
    global base_model
    logger.info(f"Cambiando modelo. Archivos LoRA recibidos: {lora_files}")

    # Liberar memoria primero
    try:
        if hasattr(base_model, 'to'):
            base_model.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Advertencia al mover modelo a CPU: {e}")

    # Si no hay archivos LoRA, volver al modelo base
    if not lora_files or all(f is None for f in lora_files):
        try:
            logger.info("Limpiando y recreando modelo base...")
            
            # Eliminar modelo actual
            del base_model
            torch.cuda.empty_cache()
            
            # Recrear modelo base limpio
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            
            # Reaplicar Flash Attention
            if hasattr(base_model.config, "use_flash_attention_2"):
                base_model.config.use_flash_attention_2 = True
            if hasattr(base_model, "text_encoder") and hasattr(base_model.text_encoder.config, "use_flash_attention_2"):
                base_model.text_encoder.config.use_flash_attention_2 = False
                
            logger.info("✅ Modelo base recreado exitosamente")
            
        except Exception as e:
            logger.error(f"Error recreando modelo base: {e}")
            # Fallback: cargar modelo base de nuevo
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
        
        active_model = base_model
        status = "✅ Modelo Base Activo (limpio)"
        new_state = {"model": base_model, "lora_path": None}
        
    else:
        # Cargar LoRA
        try:
            import tempfile
            import shutil
            
            # Crear directorio temporal
            lora_temp_dir = tempfile.mkdtemp()
            logger.info(f"Directorio temporal para LoRA: {lora_temp_dir}")
            
            # Copiar archivos
            for file_path in lora_files:
                if file_path and file_path != "None":
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(lora_temp_dir, filename)
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copiado: {filename}")
            
            # Verificar archivos requeridos
            required_files = {"adapter_config.json", "adapter_model.safetensors"}
            files_in_dir = set(os.listdir(lora_temp_dir))
            
            if not required_files.issubset(files_in_dir):
                missing = required_files - files_in_dir
                raise ValueError(f"Faltan archivos LoRA: {missing}")
            
            logger.info(f"Archivos en directorio temporal: {files_in_dir}")
            
            # Limpiar modelo base si tiene LoRA cargado
            if hasattr(base_model, 'peft_config'):
                logger.info("Limpiando modelo base antes de cargar LoRA...")
                del base_model
                torch.cuda.empty_cache()
                base_model = MusicgenForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                )
            
            # Cargar LoRA
            active_model = PeftModel.from_pretrained(base_model, lora_temp_dir)
            logger.info("✅ LoRA cargado exitosamente")
            
            # Aplicar Flash Attention al modelo LoRA
            if hasattr(active_model.config, "use_flash_attention_2"):
                active_model.config.use_flash_attention_2 = True
            if hasattr(active_model, "text_encoder") and hasattr(active_model.text_encoder.config, "use_flash_attention_2"):
                active_model.text_encoder.config.use_flash_attention_2 = False
            
            status = "✅ LoRA activo"
            new_state = {"model": active_model, "lora_path": lora_temp_dir}
            
        except Exception as e:
            logger.error(f"Error cargando LoRA: {e}")
            
            # Limpiar directorio temporal
            if 'lora_temp_dir' in locals():
                shutil.rmtree(lora_temp_dir, ignore_errors=True)
            
            # Fallback a modelo base
            try:
                base_model = MusicgenForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                )
            except Exception as fallback_error:
                logger.error(f"Error en fallback a modelo base: {fallback_error}")
                # Último fallback - devolver el modelo base actual
                pass
                
            active_model = base_model
            status = f"❌ Error al cargar LoRA: {str(e)[:100]}..."
            new_state = {"model": base_model, "lora_path": None}
    
    # Mover modelo a CPU y limpiar cache
    try:
        active_model.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Advertencia al mover modelo a CPU después de carga: {e}")
    
    logger.info(status)
    return new_state, status


def generate_music_with_state(
    current_state: Dict[str, Any],
    lora_path_from_file_input: List[str],
    prompt: str,
    duration: int,
    seed: int,
    guidance: float,
    temp: float,
    topk: int,
    topp: float,
) -> Tuple[Dict[str, Any], str, Tuple[int, Any]]:
    """
    Genera audio gestionando el estado del modelo y cambiando LoRA si es necesario.
    """
    logger.info("Iniciando generación de audio...")

    # Inicializar active_model con el modelo del estado actual
    active_model = current_state["model"]
    status = "Usando modelo actual"

    # Detectar si queremos volver al modelo base
    want_base_model = (not lora_path_from_file_input or 
                      all(f is None for f in lora_path_from_file_input))
    
    # Detectar si actualmente tenemos LoRA cargado
    has_lora_loaded = current_state["lora_path"] is not None

    # Forzar cambio si es necesario
    if want_base_model and has_lora_loaded:
        logger.info("Forzando cambio a modelo base...")
        new_state, status = switch_model_and_state([])
        active_model = new_state["model"]
        current_state = new_state
    elif not want_base_model and not has_lora_loaded:
        new_state, status = switch_model_and_state(lora_path_from_file_input)
        active_model = new_state["model"]
        current_state = new_state
    elif not want_base_model and has_lora_loaded:
        current_files = set()
        if current_state["lora_path"] and os.path.exists(current_state["lora_path"]):
            current_files = set(os.listdir(current_state["lora_path"]))
        
        new_files = set()
        for file_path in lora_path_from_file_input:
            if file_path and file_path != "None":
                new_files.add(os.path.basename(file_path))
        
        if new_files != current_files:
            new_state, status = switch_model_and_state(lora_path_from_file_input)
            active_model = new_state["model"]
            current_state = new_state
        else:
            status = f"✅ LoRA activo"
            active_model = current_state["model"]
    else:
        status = "✅ Modelo Base Activo"
        active_model = current_state["model"]

    # Mover modelo a GPU
    active_model = active_model.to("cuda")
    logger.info("Modelo movido a GPU")

    # Semilla
    if seed is not None and int(seed) != -1:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        logger.info(f"Semilla fijada: {seed}")

    # Tokenizar prompt
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    logger.info(f"Prompt procesado: {prompt}")

    # Limitar tokens para evitar artefactos
    max_tokens = min(int(duration * 50), 2048)
    logger.info(f"Duración solicitada: {duration}s → max_new_tokens: {max_tokens}")

    # Generar audio
    with torch.no_grad():
        audio_codes = active_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=guidance,
            temperature=temp,
            top_k=int(topk) if topk > 0 else 250,
            top_p=topp if topp > 0 else None,
        )

    logger.info(f"Audio codes generado. Forma: {audio_codes.shape}")

    # Mover modelo de vuelta a CPU
    active_model = active_model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("Modelo movido a CPU y VRAM liberada")

    # Conversión correcta del tensor a audio
    try:
        # Para MusicGen, el tensor generado ya es la waveform
        if audio_codes.dim() == 3:  # (batch, channels, time)
            audio_np = audio_codes[0].cpu().numpy()  # (channels, time)
            if audio_np.shape[0] <= 2:  # stereo o mono
                audio_np = audio_np.T  # (time, channels)
            else:
                audio_np = audio_np[0]  # mono
        elif audio_codes.dim() == 2:  # (batch, time) o (channels, time)
            audio_np = audio_codes[0].cpu().numpy() if audio_codes.shape[0] == 1 else audio_codes.cpu().numpy()
            if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
                audio_np = audio_np.T
        else:
            audio_np = audio_codes.cpu().numpy()
            
    except Exception as e:
        logger.error(f"Error en conversión de audio: {e}")
        # Fallback seguro
        audio_np = np.zeros((32000 * duration,))  # array de silencio

    sr = 32000  # MusicGen usa 32kHz

    logger.info(f"Audio final convertido. Forma: {audio_np.shape}")
    logger.info("✅ Generación completada correctamente")
    return current_state, status, (sr, audio_np)


# --------------------------------------------------------------------------- #
# INTERFAZ GRADIO ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎶 Interfaz de Entrenamiento y Generación – MusicGen v4.6")

    initial_state = {"model": base_model, "lora_path": None}
    active_model_state = gr.State(value=initial_state)

    with gr.Tabs() as tabs:
        with gr.TabItem("🛠️ Entrenar LoRA"):
            with gr.Row():
                prep_dataset_path_input = gr.Textbox(
                    label="📂 Ruta a la carpeta con tus audios",
                    value=settings.get("dataset_path", ""),
                )
                generate_metadata_button = gr.Button(
                    "🤖 Generar `metadata.jsonl` (con captioning.py)"
                )
                # === AÑADIR ESTO PARA AUGMENTACIÓN ===
                augment_dataset_btn = gr.Button(
                    "🔄 Augmentar Dataset"
                )
                # === HASTA AQUÍ ===
                metadata_output = gr.Textbox(
                    label="Resultado", lines=2, interactive=False
                )
            # === AÑADIR ESTE BLOQUE PARA LOS NUEVOS CONTROLES ===
            with gr.Row():
                augmented_output_path = gr.Textbox(
                    label="📂 Ruta de salida para dataset augmentado",
                    value="./augmented_training_data",
                )
                use_augmented_cb = gr.Checkbox(
                    label="Usar dataset augmentado para entrenamiento",
                    value=False
                )    
            with gr.Row():
                output_dir_input = gr.Textbox(
                    label="📁 Carpeta de salida (LoRA)",
                    value=settings.get("output_dir", ""),
                )
                epochs_input = gr.Slider(
                    label="Épocas", minimum=1, maximum=500, step=1, value=settings.get("epochs", 15)
                )
                lr_input = gr.Number(
                label="Learning Rate",
                    value=settings.get("lr", 0.0001),
                    precision=6
                )
            with gr.Row():
                max_duration_input = gr.Slider(
                    label="Duración máx. del audio (s)",
                    minimum=10, maximum=300, step=1, value=settings.get("max_duration", 180),
                )
                r_input = gr.Slider(
                    label="R (rank)", minimum=4, maximum=128, step=4, value=settings.get("lora_r", 32)
                )
                alpha_input = gr.Slider(
                    label="Alpha", minimum=4, maximum=256, step=4, value=settings.get("lora_alpha", 64)
                )
            train_seed_input = gr.Number(
                label="Semilla (entrenamiento)", value=settings.get("train_seed", 42), precision=0,
            )
            launch_train_btn = gr.Button("🚀 Lanzar entrenamiento", variant="primary")
            train_log = gr.Textbox(label="Log del entrenamiento", lines=15, interactive=False)

        with gr.TabItem("✍️ Gestor de Prompts"):
            with gr.Row():
                prompt_select_dd = gr.Dropdown(
                    label="Prompts guardados", choices=prompt_manager.get_prompt_names()
                )
                prompt_name_tb = gr.Textbox(label="Nombre del Prompt (para guardar)")
                save_prompt_btn = gr.Button("💾 Guardar/Actualizar")
                delete_prompt_btn = gr.Button("🗑️ Eliminar")
                prompt_status_tb = gr.Textbox(label="Estado", interactive=False)
            prompt_text_area = gr.Textbox(label="Texto del Prompt", lines=10)
            with gr.Row():
                ollama_model_dd = gr.Dropdown(
                    label="Modelo Ollama",
                    choices=available_ollama_models, value=settings.get("ollama_model", ""),
                )
                unload_ollama_btn = gr.Button("🗑️ Descargar modelo Ollama")
                use_captions_cb = gr.Checkbox(
                    label="Usar captions del dataset como contexto",
                    info="Lee `metadata.jsonl` de la carpeta del dataset.",
                )
                enhance_btn = gr.Button("🔧 Mejorar con Ollama", variant="primary")
            use_in_inference_btn = gr.Button("🎵 Usar este Prompt en el Generador")

        with gr.TabItem("🎵 Generador (Inferencia)"):
            with gr.Row():
                prompt_input = gr.Textbox(
                    label="Prompt musical",
                    placeholder="Ej: Un solo de piano clásico...",
                    value=settings.get("inference_prompt", ""), lines=2,
                )
                # ---- INICIO DE LA MODIFICACIÓN ----
                # Ahora acepta múltiples archivos (file_count="multiple")
                lora_path_input = gr.File(
                    label="📂 Arrastra AQUÍ AMBOS archivos LoRA (config y safetensors)",
                    type="filepath",
                    file_count="multiple"
                )
                # ---- FIN DE LA MODIFICACIÓN ----
                inference_seed_input = gr.Number(
                    label="Semilla (-1 = aleatoria)",
                    value=settings.get("inference_seed", -1), precision=0,
                )
                generate_btn = gr.Button("🎹 Generar", variant="primary")
                status_output = gr.Textbox(
                    label="Modelo activo", interactive=False, value="✅ Modelo Base Activo"
                )
            duration_slider = gr.Slider(
                label="Duración (s)", minimum=5, maximum=60, step=1,
                value=settings.get("inference_duration", 15),
            )
            with gr.Accordion("Ajustes avanzados", open=False):
                guidance_slider = gr.Slider(
                    label="Guidance Scale (CFG)", minimum=1.0, maximum=20.0,
                    step=0.5, value=settings.get("guidance_scale", 3.0),
                )
                temperature_slider = gr.Slider(
                    label="Temperatura", minimum=0.1, maximum=2.0,
                    step=0.05, value=settings.get("temperature", 1.0),
                )
                topk_slider = gr.Slider(
                    label="Top‑k (0 = default 250)", minimum=0, maximum=500,
                    step=10, value=settings.get("top_k", 250),
                )
                topp_slider = gr.Slider(
                    label="Top‑p (0 = desactivado)", minimum=0.0, maximum=1.0,
                    step=0.05, value=settings.get("top_p", 0.0),
                )
            audio_out = gr.Audio(label="Resultado", type="numpy")
                        # === AÑADE ESTO: Botón para guardar el audio ===
            with gr.Row():
                save_audio_btn = gr.Button("💾 Guardar Audio Generado")
                save_output = gr.Textbox(label="Estado del guardado", interactive=False)
            
            # === Y ESTO: Conectar el botón a la función ===
            save_audio_btn.click(
                fn=save_generated_audio,
                inputs=audio_out,
                outputs=save_output
            )

    def on_select_prompt(name):
        return prompt_manager.get_prompt_text(name)
    def on_save_prompt(name, text):
        msg = prompt_manager.update_prompt(name, text)
        return msg, gr.Dropdown(choices=prompt_manager.get_prompt_names(), value=name)
    def on_delete_prompt(name):
        msg = prompt_manager.delete_prompt(name)
        return msg, gr.Dropdown(choices=prompt_manager.get_prompt_names(), value=None)

    prompt_select_dd.change(fn=on_select_prompt, inputs=prompt_select_dd, outputs=prompt_text_area)
    save_prompt_btn.click(fn=on_save_prompt, inputs=[prompt_name_tb, prompt_text_area], outputs=[prompt_status_tb, prompt_select_dd])
    delete_prompt_btn.click(fn=on_delete_prompt, inputs=prompt_name_tb, outputs=[prompt_status_tb, prompt_select_dd])
    unload_ollama_btn.click(fn=ollama.unload_ollama_model, inputs=ollama_model_dd, outputs=prompt_status_tb)
    enhance_btn.click(fn=ollama.enhance_and_translate_prompt, inputs=[ollama_model_dd, prompt_text_area, use_captions_cb, prep_dataset_path_input], outputs=prompt_text_area)
    use_in_inference_btn.click(fn=lambda txt: (txt, gr.Tabs(selected=2)), inputs=prompt_text_area, outputs=[prompt_input, tabs])
    generate_metadata_button.click(fn=generate_metadata, inputs=prep_dataset_path_input, outputs=metadata_output)
    launch_train_btn.click(fn=modify_and_run_training, inputs=[prep_dataset_path_input, output_dir_input, epochs_input, lr_input, r_input, alpha_input, max_duration_input, train_seed_input, use_augmented_cb, augmented_output_path], outputs=train_log)

    lora_path_input.change(fn=switch_model_and_state, inputs=lora_path_input, outputs=[active_model_state, status_output])

    generate_btn.click(fn=generate_music_with_state,
        inputs=[
            active_model_state, lora_path_input, prompt_input, duration_slider,
            inference_seed_input, guidance_slider, temperature_slider, topk_slider, topp_slider,
        ],
        outputs=[active_model_state, status_output, audio_out]
    )
    augment_dataset_btn.click(
        fn=augment_dataset_simple,
        inputs=[prep_dataset_path_input, augmented_output_path],
        outputs=metadata_output
    )

    def _persist(key: str, val: Any) -> None:
        settings[key] = val
        save_settings(settings)

    component_key_map = [
        (prep_dataset_path_input, "dataset_path"), (output_dir_input, "output_dir"),
        (epochs_input, "epochs"), (lr_input, "lr"), (max_duration_input, "max_duration"),
        (r_input, "lora_r"), (alpha_input, "lora_alpha"), (train_seed_input, "train_seed"),
        (prompt_input, "inference_prompt"), (duration_slider, "inference_duration"),
        (lora_path_input, "lora_path"), (inference_seed_input, "inference_seed"),
        (guidance_slider, "guidance_scale"), (temperature_slider, "temperature"),
        (topk_slider, "top_k"), (topp_slider, "top_p"),
    ]
    for comp, key in component_key_map:
        comp.change(fn=lambda v, k=key: _persist(k, v), inputs=comp, outputs=None)

if __name__ == "__main__":
    demo.launch(share=False)