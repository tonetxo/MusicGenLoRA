# -*- coding: utf-8 -*-
import json
import logging
import os
import re
import shutil
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
from captioning import AudioTagger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
SETTINGS_FILE = "settings.json"
MODEL_ID = "facebook/musicgen-small"

# === UTILS ===
def load_settings():
    defaults = {
        "dataset_path": "",
        "output_dir": "./mi_lora_final",
        "epochs": 15,
        "lr": 0.0001,
        "lora_r": 8,
        "lora_alpha": 16,
        "max_duration": 8,
        "ollama_model": "",
        "train_seed": 42,
        "inference_prompt": "tonetxo_style, synthwave",
        "inference_duration": 8,
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
        except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
            logger.warning(f"Could not load settings file {SETTINGS_FILE}: {e}")
        except Exception as e:
            logger.warning(f"Unexpected error loading settings: {e}")
    return defaults

def save_settings(settings):
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

settings = load_settings()

# === MODELO BASE ===
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
base_model.to("cpu")

# === CLASES ===
class PromptManager:
    def __init__(self, prompts_file="saved_prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()
    def load_prompts(self):
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, "r", encoding="utf-8") as f:
                    loaded_prompts = json.load(f)
                    # Ensure all prompts have "tonetxo_style" in them
                    for prompt in loaded_prompts:
                        if "tonetxo_style" not in prompt["text"].lower():
                            prompt["text"] = f"tonetxo_style, {prompt['text']}"
                    return loaded_prompts
            except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError) as e:
                logger.warning(f"Could not load prompts file {self.prompts_file}: {e}")
                return []
        return []
    def save_prompts(self):
        with open(self.prompts_file, "w", encoding="utf-8") as f:
            json.dump(self.prompts, f, indent=4, ensure_ascii=False)
    def update_prompt(self, name, text):
        # Ensure tonetxo_style is included in the prompt
        if "tonetxo_style" not in text.lower():
            text = f"tonetxo_style, {text}"
        
        for p in self.prompts:
            if p["name"] == name:
                p["text"] = text
                break
        else:
            self.prompts.append({"name": name, "text": text, "created": datetime.now().isoformat()})
        self.save_prompts()
        return f"Prompt '{name}' guardado."
    def delete_prompt(self, name):
        before = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["name"] != name]
        if len(self.prompts) < before:
            self.save_prompts()
            return f"Prompt '{name}' eliminado."
        return "No encontrado."
    def get_prompt_text(self, name): return next((p["text"] for p in self.prompts if p["name"] == name), "")
    def get_prompt_names(self): return sorted([p["name"] for p in self.prompts])

class OllamaPromptManager:
    def __init__(self, prompts_file="ollama_prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()
        
        # Add default templates if file doesn't exist or is empty
        if not self.prompts:
            self.add_default_templates()
    
    def add_default_templates(self):
        """Add default templates for users to get started"""
        default_templates = [
            {
                "name": "Predeterminado",
                "template": "You are an expert music-prompt writer. Always ensure 'tonetxo_style' is included in the generated prompt. Improve and translate to English for MusicGen: \\\"{prompt}\\\"",
                "created": datetime.now().isoformat(),
                "default": True
            },
            {
                "name": "Creativo",
                "template": "As a creative music prompt engineer, transform the following natural language prompt into an inspiring, detailed description for MusicGen AI. Make it more vivid and descriptive, and ensure 'tonetxo_style' is included: \\\"{prompt}\\\". Focus on rich textures and emotional depth.",
                "created": datetime.now().isoformat(),
                "default": True
            },
            {
                "name": "Técnico",
                "template": "You are a technical music production expert. Analyze and enhance the following prompt for MusicGen, focusing on technical aspects like instrumentation, genre, and production quality. Ensure 'tonetxo_style' is included in the output. Prompt: \\\"{prompt}\\\". Provide a technically detailed, professional music generation instruction.",
                "created": datetime.now().isoformat(),
                "default": True
            }
        ]
        
        self.prompts = default_templates
        self.save_prompts()
    
    def load_prompts(self):
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except (json.JSONDecodeError, UnicodeDecodeError, FileNotFoundError):
                pass
        return []
    
    def save_prompts(self):
        with open(self.prompts_file, "w", encoding="utf-8") as f:
            json.dump(self.prompts, f, indent=4, ensure_ascii=False)
    
    def update_prompt(self, name, template):
        for p in self.prompts:
            if p["name"] == name:
                p["template"] = template
                break
        else:
            self.prompts.append({
                "name": name, 
                "template": template, 
                "created": datetime.now().isoformat(),
                "default": False  # Mark as user-defined
            })
        self.save_prompts()
        return f"Plantilla Ollama '{name}' guardada."
    
    def delete_prompt(self, name):
        before = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["name"] != name]
        if len(self.prompts) < before:
            self.save_prompts()
            return f"Plantilla Ollama '{name}' eliminada."
        return "No encontrado."
    
    def get_prompt_template(self, name):
        for p in self.prompts:
            if p["name"] == name:
                return p["template"]
        return ""
    
    def get_prompt_names(self):
        return sorted([p["name"] for p in self.prompts])

class OllamaIntegration:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    def get_available_models(self):
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()  # Raises an HTTPError for bad responses
            return [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return ["Error: No se pudo conectar a Ollama"]
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Ollama response: {e}")
            return ["Error: No se pudo conectar a Ollama"]
    def enhance_and_translate_prompt(self, model_name, base_prompt, use_captions, dataset_path, prompt_template=None):
        if not base_prompt.strip(): return "Prompt vacío"
        
        # Process the prompt to ensure it includes tonetxo_style
        processed_prompt = base_prompt
        if "tonetxo_style" not in processed_prompt.lower():
            processed_prompt = f"tonetxo_style, {processed_prompt}"
        
        # Use the template if provided, otherwise use default
        if prompt_template:
            # Replace {prompt} placeholder in the template with the actual prompt
            ollama_prompt = prompt_template.replace("{prompt}", processed_prompt)
        else:
            # Default template if no custom template is provided
            ollama_prompt = f"You are an expert music-prompt writer. Always ensure 'tonetxo_style' is included in the generated prompt. Improve and translate to English for MusicGen: \\\"{processed_prompt}\\\""
        
        try:
            r = self.session.post(f"{self.base_url}/api/generate", json={"model": model_name, "prompt": ollama_prompt, "stream": False}, timeout=60)
            r.raise_for_status()
            out = r.json().get("response", "").strip().replace('"', "")
            # Ensure tonetxo_style is included in the output
            if "tonetxo_style" not in out.lower():
                out = f"tonetxo_style, {out}"
            return out[0].upper() + out[1:] if out else processed_prompt
        except requests.exceptions.RequestException as e:
            logger.error(f"Error connecting to Ollama: {e}")
            return "Error en Ollama"
        except (KeyError, ValueError) as e:
            logger.error(f"Error parsing Ollama response: {e}")
            return "Error en Ollama"
    def unload_ollama_model(self, model_name):
        try:
            result = subprocess.run(
                ["ollama", "stop", model_name],
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode == 0:
                return f"✅ Modelo Ollama '{model_name}' descargado de la memoria."
            else:
                return f"⚠️ No se pudo descargar el modelo '{model_name}': {result.stderr}"
        except FileNotFoundError:
            return "❌ Comando 'ollama' no encontrado."
        except Exception as e:
            return f"⚠️ Error al descargar modelo Ollama: {str(e)}"

def free_gpu_memory():
    global base_model
    try:
        if base_model is not None:
            base_model.to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            return "✅ Memoria GPU liberada correctamente."
        else:
            return "ℹ️ No hay GPU activa para liberar."
    except Exception as e:
        return f"❌ Error liberando memoria GPU: {str(e)}"

prompt_manager = PromptManager()
ollama = OllamaIntegration()
ollama_prompt_manager = OllamaPromptManager()  # New manager for Ollama prompt templates
available_ollama_models = ollama.get_available_models()
if not settings["ollama_model"] and available_ollama_models:
    settings["ollama_model"] = available_ollama_models[0]

# === AUDIO TAGGER ===
tagger = AudioTagger()
tagger_loaded = False
try:
    msg = tagger.load_model()
    if "Error" not in msg: tagger_loaded = True
except Exception as e:
    logger.warning(f"Could not load AudioTagger model: {e}")
    pass

def _extract_description(raw, audio_path=None):
    result = {
        "description": "electronic music, synth, drums", 
        "bpm": "", 
        "genre": "electronic", 
        "moods": []
    }
    desc = ""
    if isinstance(raw, dict):
        if "caption" in raw and raw["caption"]:
            desc = raw["caption"]
        elif "labels" in raw and raw["labels"]:
            labels = raw["labels"]
            if labels:
                main_labels = [label.get("cleaned_label", label["label"]) 
                             for label in labels[:4] if label.get("confidence", 0) > 0.1]
                if main_labels:
                    desc = ", ".join(main_labels)
    elif isinstance(raw, str):
        desc = raw
    if not desc or desc == "audio caption placeholder":
        desc = "electronic music, synth, drums"
    if "tonetxo_style" not in desc.lower():
        desc = f"tonetxo_style, {desc}"
    result["description"] = desc
    if audio_path and os.path.exists(audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result["bpm"] = str(round(float(tempo))) if tempo > 0 else ""
        except Exception as e:
            logger.warning(f"No se pudo calcular BPM para {audio_path}: {e}")
    return result

def generate_metadata(dataset_dir, min_duration=5.0, max_duration=60.0):
    if not tagger_loaded: return "❌ AudioTagger no disponible"
    root = Path(dataset_dir)
    if not root.is_dir(): return f"❌ Ruta inválido: {dataset_dir}"
    exts = {".wav", ".mp3", ".flac"}
    audio_files = [p for p in root.iterdir() if p.suffix.lower() in exts]
    if not audio_files: return "⚠️ No hay audios"
    
    # Remove old metadata.jsonl if it exists to avoid interference
    old_metadata_path = root / "metadata.jsonl"
    if old_metadata_path.exists():
        try:
            old_metadata_path.unlink()  # Remove the old file
            logger.info(f"Removed old metadata file: {old_metadata_path}")
        except OSError as e:
            logger.warning(f"Could not remove old metadata file {old_metadata_path}: {e}")
    
    out_path = root / "metadata.jsonl"
    with out_path.open("w", encoding="utf-8") as out_f:
        for audio_path in tqdm.tqdm(audio_files, desc="Generando metadata"):
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = float(librosa.get_duration(y=y, sr=sr))
                
                # Skip files that are too short or too long
                if duration < min_duration or duration > max_duration:
                    logger.info(f"Skipping {audio_path.name} (duration: {duration:.2f}s) - outside range [{min_duration}s, {max_duration}s]")
                    continue
                    
            except:
                sr, duration = 44100, 0.0
            
            try:
                raw_res = tagger.process_audio_file(str(audio_path))
            except Exception as e:
                logger.warning(f"Error processing audio file for captioning: {e}")
                raw_res = {}
            
            audio_info = _extract_description(raw_res, str(audio_path))
            raw_desc = audio_info["description"].replace("Audio containing ", "")
            full_desc = raw_desc
            caption_dict = {
                "key": "",
                "artist": "Tonetxo",
                "sample_rate": int(sr),
                "file_extension": audio_path.suffix.lstrip("."),
                "description": full_desc,
                "keywords": "",
                "duration": round(duration, 2),
                "bpm": audio_info["bpm"],
                "genre": audio_info["genre"],
                "title": audio_path.stem.replace("_", " ").title(),
                "name": audio_path.stem,
                "instrument": "Mix",
                "moods": audio_info["moods"],
                "file_name": audio_path.name,
                "audio_filepath": str(audio_path.resolve())
            }
            out_f.write(json.dumps(caption_dict, ensure_ascii=False) + "\n")
    return f"✅ metadata.jsonl creado en: {out_path} (filtered to [{min_duration}s, {max_duration}s])"

# === CHUNKING ===
def chunk_dataset(input_dataset_path: str, output_dataset_path: str, chunk_duration: float = 30.0) -> str:
    """Divide archivos de audio largos en segmentos más pequeños para entrenamiento"""
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)
    chunked_audio_dir = output_path / "chunked_audio"
    
    if not input_path.is_dir():
        return f"❌ Directorio de entrada inválido: {input_dataset_path}"
    
    # Get all audio files in the input directory
    exts = {".wav", ".mp3", ".flac"}
    audio_files = [p for p in input_path.iterdir() if p.suffix.lower() in exts]
    if not audio_files:
        return f"❌ No se encontraron archivos de audio en {input_dataset_path}"
    
    # Clean up old chunked dataset if it exists to avoid interference
    if output_path.exists() and output_path.is_dir():
        try:
            shutil.rmtree(output_path)
            logger.info(f"Removed old chunked dataset: {output_path}")
        except OSError as e:
            logger.warning(f"Could not remove old chunked dataset {output_path}: {e}")
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        chunked_audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        return f"❌ Error creando directorio de salida: {e}"
    
    total_chunks = 0
    new_metadata_path = output_path / "metadata.jsonl"
    
    try:
        with open(new_metadata_path, 'w', encoding='utf-8') as f_out:
            for audio_file in tqdm.tqdm(audio_files, desc="Procesando archivos"):
                try:
                    # Load the audio file to get its duration
                    y, sr = librosa.load(str(audio_file), sr=None, mono=True)
                    total_duration = librosa.get_duration(y=y, sr=sr)
                    
                    # Calculate number of chunks needed based on specified duration
                    num_chunks = max(1, int(total_duration // chunk_duration) + (1 if total_duration % chunk_duration > 0 else 0))
                    
                    for i in range(num_chunks):
                        start_time = i * chunk_duration
                        end_time = min((i + 1) * chunk_duration, total_duration)
                        
                        # Extract the chunk
                        start_sample = int(start_time * sr)
                        end_sample = int(end_time * sr)
                        chunk_y = y[start_sample:end_sample]
                        
                        # Create filename for the chunk
                        chunk_filename = f"{audio_file.stem}_chunk_{i+1:03d}.wav"
                        chunk_path = chunked_audio_dir / chunk_filename
                        
                        # Save the chunk
                        sf.write(str(chunk_path), chunk_y, sr)
                        
                        # To avoid AudioTagger's internal chunking (which uses 10s chunks), 
                        # process a larger chunk if needed for description or use original file's description
                        # First, try to get description from original file if it's the first chunk
                        if i == 0:  # Get description from original file for the first chunk
                            original_raw_res = {}
                            if tagger_loaded:
                                try:
                                    original_raw_res = tagger.process_audio_file(str(audio_file))
                                except Exception as e:
                                    logger.warning(f"Error processing original file for captioning: {e}")
                            
                            audio_info = _extract_description(original_raw_res, str(audio_file))
                        else:  # For subsequent chunks, use a simple approach to avoid AudioTagger's internal chunking
                            # Just reuse the description from the original file
                            # Since all chunks from the same file should have similar characteristics
                            audio_info = {
                                "description": "tonetxo_style, electronic music, synth",  # Default with tonetxo_style
                                "bpm": "", 
                                "genre": "electronic", 
                                "moods": []
                            }
                        
                        raw_desc = audio_info["description"].replace("Audio containing ", "")
                        full_desc = raw_desc
                        
                        # Calculate actual chunk duration
                        chunk_actual_duration = end_time - start_time
                        
                        # Create metadata entry for this chunk
                        chunk_metadata = {
                            "key": "",
                            "artist": "Tonetxo",
                            "sample_rate": int(sr),
                            "file_extension": "wav",
                            "description": full_desc,
                            "keywords": "",
                            "duration": round(chunk_actual_duration, 2),
                            "bpm": audio_info["bpm"],
                            "genre": audio_info["genre"],
                            "title": f"{audio_file.stem} Chunk {i+1}",
                            "name": f"{audio_file.stem}_chunk_{i+1:03d}",
                            "instrument": "Mix",
                            "moods": audio_info["moods"],
                            "file_name": chunk_filename,
                            "audio_filepath": str(chunk_path.resolve())
                        }
                        
                        f_out.write(json.dumps(chunk_metadata, ensure_ascii=False) + '\n')
                        total_chunks += 1
                
                except Exception as e:
                    logger.error(f"Error procesando archivo {audio_file}: {e}")
                    continue
        
        return f"✅ Dataset troceado: {total_chunks} segmentos de ~{chunk_duration}s en {output_path}"
    
    except Exception as e:
        return f"❌ Error en troceado: {str(e)}"


# === POST-PROCESAMIENTO DE DESCRIPCIONES ===
def postprocess_descriptions(dataset_path: str, use_ollama: bool = True, ollama_model: str = None) -> str:
    """Mejora las descripciones en metadata.jsonl para mayor consistencia y riqueza"""
    import json
    from pathlib import Path
    import time
    
    metadata_path = Path(dataset_path) / "metadata.jsonl"
    if not metadata_path.exists():
        return f"❌ No se encontró metadata.jsonl en {dataset_path}"
    
    # Leer el metadata actual
    entries = []
    with open(metadata_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    if not entries:
        return "❌ No hay entradas en el metadata para procesar"
    
    yield f"📊 Cargando {len(entries)} entradas de metadata..."
    
    # Agrupar entradas por archivo original (basado en el nombre base)
    grouped_entries = {}
    for entry in entries:
        # Extraer el nombre base del archivo original del chunk
        original_name = "_".join(entry["name"].split("_")[:-2]) if "_chunk_" in entry["name"] else entry["name"]
        if original_name not in grouped_entries:
            grouped_entries[original_name] = []
        grouped_entries[original_name].append(entry)
    
    yield f"📋 Identificados {len(grouped_entries)} grupos de archivos originales"
    
    # Procesar cada grupo de chunks del mismo archivo original
    updated_count = 0
    total_groups = len(grouped_entries)
    processed_groups = 0
    
    for original_name, chunk_entries in grouped_entries.items():
        processed_groups += 1
        yield f"🔄 Procesando grupo {processed_groups}/{total_groups}: {original_name} ({len(chunk_entries)} chunks)"
        
        # Tomar la descripción más detallada como base (generalmente del primer chunk)
        best_description = chunk_entries[0]["description"]  # Comenzamos con el primer chunk
        
        # Encontrar la descripción más detallada entre todos los chunks
        for entry in chunk_entries:
            if len(entry["description"]) > len(best_description):
                best_description = entry["description"]
        
        # Si usamos Ollama, mejorar la descripción base
        if use_ollama and ollama_model:
            try:
                # Plantilla más específica para descripciones musicales concisas
                ollama_template = "Generate only a clean MusicGen prompt in English that includes 'tonetxo_style' at the beginning. Focus on instruments, genre, mood, and production style. Keep it concise and specific: \\\"{prompt}\\\". Output only the prompt, nothing else. Do not include any explanations, translations, or formatting."
                enhanced_description = ollama.enhance_and_translate_prompt(
                    ollama_model, best_description, False, "", ollama_template
                )
                if enhanced_description and "Error" not in enhanced_description:
                    # Limpiar cualquier contenido extra que Ollama haya incluido
                    import re
                    # Buscar posibles formatos de respuesta y extraer solo la descripción
                    if "**" in enhanced_description or "Final Answer:" in enhanced_description:
                        # Extraer solo la parte principal del prompt
                        lines = enhanced_description.split('\n')
                        main_prompt = []
                        for line in lines:
                            if line.strip() and not any(marker in line for marker in ['**', 'Final Answer:', 'Option', 'Translation:', 'Justification:', '- The']):
                                if line.strip() not in ['```', '```', '```', '```']:  # Ignorar bloques de código
                                    main_prompt.append(line.strip())
                        enhanced_description = '. '.join(main_prompt).strip()
                    
                    # Asegurarse de que 'tonetxo_style' esté incluido al principio
                    if "tonetxo_style" not in enhanced_description.lower():
                        enhanced_description = f"tonetxo_style, {enhanced_description}"
                    
                    # Limpiar la descripción de posibles explicaciones adicionales
                    if "tonetxo_style, **" in enhanced_description or "**" in enhanced_description:
                        # Eliminar marcadores de formato
                        enhanced_description = re.sub(r'\*\*.*?\*\*', '', enhanced_description)  # Remover **text**
                        enhanced_description = re.sub(r'```.*?```', '', enhanced_description, flags=re.DOTALL)  # Remover bloques de código
                        enhanced_description = enhanced_description.replace("Option 1:", "").replace("Option 2:", "").replace("Option 3:", "")
                        enhanced_description = enhanced_description.replace("Translation:", "").replace("Justification:", "")
                        enhanced_description = enhanced_description.strip()
                    
                    best_description = enhanced_description
                else:
                    yield f"⚠️ Ollama no pudo mejorar la descripción para {original_name}, usando la original"
            except Exception as e:
                logger.warning(f"Error usando Ollama para mejorar descripciones: {e}")
                yield f"⚠️ Error con Ollama para {original_name}: {str(e)[:50]}..., usando original"
        
        # Aplicar la mejor descripción a todos los chunks del mismo archivo original
        for entry in chunk_entries:
            # Conservar el 'tonetxo_style' y aplicar la descripción mejorada
            if "tonetxo_style" not in best_description.lower():
                best_description = f"tonetxo_style, {best_description}"
            
            entry["description"] = best_description
            updated_count += 1
        
        # Pequeña pausa para no sobrecargar Ollama si está habilitado
        if use_ollama:
            time.sleep(0.1)
    
    # Guardar el metadata actualizado
    backup_path = metadata_path.with_suffix('.jsonl.backup')
    import shutil
    shutil.copy2(metadata_path, backup_path)  # Crear backup
    
    yield f"💾 Guardando {updated_count} descripciones actualizadas..."
    
    with open(metadata_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    yield f"✅ {updated_count} descripciones actualizadas en {dataset_path}/metadata.jsonl (backup en {backup_path})"


# === AUGMENTACIÓN ===
def augment_dataset_simple(input_dataset_path: str, output_dataset_path: str) -> str:
    input_path = Path(input_dataset_path)
    output_path = Path(output_dataset_path)
    augmented_audio_dir = output_path / "augmented_audio"
    if not input_path.is_dir():
        return f"❌ Directorio de entrada inválido: {input_dataset_path}"
    metadata_file = input_path / "metadata.jsonl"
    if not metadata_file.exists():
        return f"❌ No se encontró 'metadata.jsonl' en {input_dataset_path}"
    
    # Clean up old augmented dataset if it exists to avoid interference
    if output_path.exists() and output_path.is_dir():
        try:
            # Remove the entire output directory and recreate it
            shutil.rmtree(output_path)
            logger.info(f"Removed old augmented dataset: {output_path}")
        except OSError as e:
            logger.warning(f"Could not remove old augmented dataset {output_path}: {e}")
    
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        augmented_audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        return f"❌ Error creando directorio de salida: {e}"
    def simple_augment_audio(y, sr, filename_prefix):
        augmented_samples = []
        try:
            augmented_samples.append((librosa.effects.pitch_shift(y, sr=sr, n_steps=-1), sr, f"{filename_prefix}_pitch_down1"))
            augmented_samples.append((librosa.effects.pitch_shift(y, sr=sr, n_steps=1), sr, f"{filename_prefix}_pitch_up1"))
        except Exception as e:
            logger.warning(f"Pitch shifting failed: {e}")
            pass
        augmented_samples.append((np.clip(y * 0.9, -1.0, 1.0), sr, f"{filename_prefix}_vol_down"))
        augmented_samples.append((np.clip(y * 1.1, -1.0, 1.0), sr, f"{filename_prefix}_vol_up"))
        try:
            augmented_samples.append((librosa.effects.time_stretch(y, rate=0.9), sr, f"{filename_prefix}_stretch_slow"))
            augmented_samples.append((librosa.effects.time_stretch(y, rate=1.1), sr, f"{filename_prefix}_stretch_fast"))
        except Exception as e:
            logger.warning(f"Time stretching failed: {e}")
            pass
        try:
            noise = np.random.randn(len(y))
            y_noise = y + 0.005 * noise
            augmented_samples.append((np.clip(y_noise, -1.0, 1.0), sr, f"{filename_prefix}_noise"))
        except Exception as e:
            logger.warning(f"Noise augmentation failed: {e}")
            pass
        return augmented_samples
    total_samples = 0
    new_metadata_path = output_path / "metadata.jsonl"
    try:
        with open(metadata_file, 'r', encoding='utf-8') as f_in, \
             open(new_metadata_path, 'w', encoding='utf-8') as f_out:
            original_lines = [line.strip() for line in f_in if line.strip()]
            for line in original_lines:
                try:
                    data = json.loads(line)
                    original_audio_path = Path(data['audio_filepath'])
                    if not original_audio_path.exists():
                        continue
                    original_filename = data['file_name']
                    new_audio_path_obj = augmented_audio_dir / original_filename
                    shutil.copy2(original_audio_path, new_audio_path_obj)
                    new_data_entry = data.copy()
                    new_data_entry['audio_filepath'] = str(new_audio_path_obj.resolve())
                    f_out.write(json.dumps(new_data_entry, ensure_ascii=False) + '\n')
                    total_samples += 1
                except Exception as e:
                    logger.error(f"Error procesando muestra original: {e}")
                    continue
            for line in tqdm.tqdm(original_lines, desc="Augmentando"):
                try:
                    data = json.loads(line)
                    original_audio_path = Path(data['audio_filepath'])
                    if not original_audio_path.exists():
                        continue
                    y, sr = librosa.load(str(original_audio_path), sr=None, mono=True)
                    filename_prefix = original_audio_path.stem
                    augmented_samples = simple_augment_audio(y, sr, filename_prefix)
                    for aug_y, aug_sr, aug_name in augmented_samples:
                        new_audio_filename = f"{aug_name}.wav"
                        new_audio_path_obj = augmented_audio_dir / new_audio_filename
                        sf.write(str(new_audio_path_obj), aug_y, aug_sr)
                        new_data = data.copy()
                        new_data['name'] = aug_name
                        new_data['title'] = f"{data['title']} ({aug_name.split('_')[-1]})"
                        new_data['file_name'] = new_audio_filename
                        new_data['audio_filepath'] = str(new_audio_path_obj.resolve())
                        new_data['description'] = data['description']
                        f_out.write(json.dumps(new_data, ensure_ascii=False) + '\n')
                        total_samples += 1
                except Exception as e:
                    logger.error(f"Error procesando muestra: {e}")
                    continue
        return f"✅ Dataset augmentado: {total_samples} muestras en {output_path}"
    except Exception as e:
        return f"❌ Error en augmentación: {str(e)}"

# === ENTRENAMIENTO ===
import signal
current_training_process = None
def interrupt_training():
    global current_training_process
    if current_training_process and current_training_process.poll() is None:
        try:
            os.killpg(os.getpgid(current_training_process.pid), signal.SIGTERM)
            current_training_process = None
            return "🛑 Entrenamiento interrumpido por el usuario."
        except Exception as e:
            return f"⚠️ Error al interrumpir: {str(e)}"
    return "No hay ningún entrenamiento en curso para interrumpir."

def modify_and_run_training(
    dataset_path, output_dir, epochs, lr, scheduler, lora_r, lora_alpha, max_duration, train_seed,
    use_augmented=False, augmented_path="", weight_decay=0.01
):
    global current_training_process
    if current_training_process and current_training_process.poll() is None:
        yield "⚠️ Ya hay un entrenamiento en curso. Interrúmpelo antes de empezar uno nuevo."
        return
    try:
        output_dir_path = Path("./musicgen-dreamboothing") / output_dir
        if output_dir_path.is_dir():
            yield f"🗑️ Limpiando directorio de salida (conservando logs): {output_dir_path}"
            for item in output_dir_path.iterdir():
                if item.name == "runs":
                    continue
                if item.is_dir():
                    shutil.rmtree(item)
                else:
                    os.remove(item)
        else:
            os.makedirs(output_dir_path, exist_ok=True)
        yield f"✅ Directorio de salida limpio creado en: {output_dir_path}"
    except Exception as e:
        yield f"❌ Error al limpiar el directorio de salida: {e}"
        return
    dataset_path_to_use = augmented_path if use_augmented and augmented_path and os.path.exists(augmented_path) else dataset_path
    absolute_dataset_path = os.path.abspath(dataset_path_to_use)
    yield f"Usando dataset: {absolute_dataset_path}\n"
    yield "✅ Script de entrenamiento verificado.\n"
    command = [
        "accelerate", "launch", "dreambooth_musicgen.py",
        f"--model_name_or_path={MODEL_ID}",
        f"--dataset_name={absolute_dataset_path}",
        f"--output_dir={output_dir}",
        f"--num_train_epochs={int(epochs)}",
        "--use_lora",
        f"--lora_r={int(lora_r)}",
        f"--lora_alpha={int(lora_alpha)}",
        f"--learning_rate={lr}",
        f"--lr_scheduler_type={scheduler}",
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--fp16",
        "--text_column_name=description",
        "--target_audio_column_name=audio_filepath",
        "--train_split_name=train",
        "--overwrite_output_dir",
        "--do_train",
        f"--max_duration_in_seconds={int(max_duration)}",
        f"--target_duration={int(max_duration)}",
        f"--seed={int(train_seed)}",
        "--instance_prompt=tonetxo_style",
        "--logging_steps=1",
        "--save_steps=30",
        f"--weight_decay={weight_decay}",
        "--gradient_checkpointing",
    ]
    full_log = "🚀 Iniciando entrenamiento...\n"
    yield full_log
    process = subprocess.Popen(command, cwd="./musicgen-dreamboothing", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    current_training_process = process
    try:
        for line in iter(process.stdout.readline, ''):
            if line:
                full_log += line.rstrip() + "\n"
                # Limit log length to prevent excessive memory usage
                if len(full_log) > 50000:  # Keep last ~50k characters
                    lines = full_log.split('\n')
                    full_log = '\n'.join(lines[-200:])  # Keep last 200 lines
                yield full_log
    except Exception as e:
        full_log += f"[ERROR] {e}\n"
        yield full_log
    finally:
        process.wait()
    full_log += "\n✅ Entrenamiento finalizado." if process.returncode == 0 else f"\n❌ Error: {process.returncode}"
    current_training_process = None
    yield full_log

# === INFERENCIA ===
def switch_model_and_state(lora_files: List[str]):
    global base_model
    try:
        base_model.to("cpu")
        torch.cuda.empty_cache()
    except Exception as e:
        logger.warning(f"Error moving model to CPU or clearing cache: {e}")
    if not lora_files or all(f is None for f in lora_files):
        del base_model
        torch.cuda.empty_cache()
        base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        return {"model": base_model, "lora_path": None}, "✅ Modelo Base"
    try:
        import tempfile, shutil
        lora_temp_dir = tempfile.mkdtemp()
        for fp in lora_files:
            if fp and fp != "None":
                shutil.copy2(fp, os.path.join(lora_temp_dir, os.path.basename(fp)))
        active_model = PeftModel.from_pretrained(base_model, lora_temp_dir)
        return {"model": active_model, "lora_path": lora_temp_dir}, "✅ LoRA cargado"
    except Exception as e:
        logger.error(f"Error LoRA: {e}")
        base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
        return {"model": base_model, "lora_path": None}, f"❌ Error LoRA: {str(e)[:100]}"

def normalize_audio_for_output(audio_data):
    """Normalize audio data for output as WAV file."""
    if audio_data.ndim == 2 and audio_data.shape[0] <= 2:
        # If stereo channels are in rows, transpose to columns
        audio_data = audio_data.T
    elif audio_data.ndim > 2:
        # If there are multiple channels, take the first one
        audio_data = audio_data[0] if audio_data.shape[0] > 1 else audio_data.squeeze(0)
    
    # Convert to int16 format for WAV
    audio_data = np.clip(audio_data, -1.0, 1.0)  # Ensure values are in [-1, 1]
    audio_data = (audio_data * 32767).astype(np.int16)
    return audio_data

def ensure_tonetxo_style(prompt):
    """Ensure the tonetxo_style tag is present in the prompt for proper metadata interpretation"""
    if "tonetxo_style" not in prompt.lower():
        prompt = f"tonetxo_style, {prompt}"
    return prompt

def generate_music_with_state(
    current_state, lora_path_from_file_input, prompt, duration, seed, guidance, temp, topk, topp
):
    # Ensure tonetxo_style is present in the prompt
    prompt = ensure_tonetxo_style(prompt)
    
    active_model = current_state["model"]
    want_base = not lora_path_from_file_input or all(f is None for f in lora_path_from_file_input)
    has_lora = current_state["lora_path"] is not None
    if (want_base and has_lora) or (not want_base and not has_lora):
        new_state, _ = switch_model_and_state(lora_path_from_file_input)
        active_model = new_state["model"]
        current_state = new_state
    active_model = active_model.to("cuda")
    if seed != -1:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    inputs = processor(text=[prompt], return_tensors="pt").to("cuda")
    
    # Calculate max tokens: 50 tokens per second of audio (MusicGen specific)
    MAX_TOKENS_PER_SECOND = 50
    MAX_TOTAL_TOKENS = 2048
    max_tokens = min(int(duration * MAX_TOKENS_PER_SECOND), MAX_TOTAL_TOKENS)
    
    with torch.no_grad():
        audio_codes = active_model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            guidance_scale=guidance,
            temperature=temp,
            top_k=topk if topk > 0 else 250,
            top_p=topp if topp > 0 else None,
        )
    active_model = active_model.to("cpu")
    torch.cuda.empty_cache()
    audio_np = audio_codes[0].cpu().numpy()
    audio_np = normalize_audio_for_output(audio_np)
    return current_state, "✅ Generado", (32000, audio_np)

def save_generated_audio(audio_data, output_dir="./generated_audio"):
    if not audio_data: 
        return "❌ No hay audio"
    try:
        sr, audio_np = audio_data
    except (ValueError, TypeError):
        return "❌ Formato de audio inválido"
    
    if audio_np is None or len(audio_np) == 0: 
        return "❌ Audio vacío"
    
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"musicgen_{timestamp}.wav")
    
    try:
        # Ensure audio has proper format for saving
        if audio_np.ndim == 2:
            # Limit to stereo (2 channels max)
            if audio_np.shape[1] > 2:
                audio_np = audio_np[:, :2]
            elif audio_np.shape[0] > 2:  # If channels are in rows
                audio_np = audio_np[:2, :].T
        # Convert to appropriate format if needed
        if audio_np.dtype != np.int16:
            audio_np = (audio_np * 32767).astype(np.int16)
        
        sf.write(filepath, audio_np, sr)
        return f"✅ Guardado: {filepath}"
    except Exception as e:
        logger.error(f"Error saving audio file: {e}")
        return f"❌ Error: {e}"

def save_all_settings(dataset_path, output_dir, epochs, lr, scheduler, weight_decay, max_duration, lora_r, lora_alpha, train_seed, inference_prompt, inference_duration, inference_seed, guidance, temp, topk, topp, ollama_model):
    settings_to_save = {
        "dataset_path": dataset_path,
        "output_dir": output_dir,
        "epochs": epochs,
        "lr": lr,
        "lr_scheduler": scheduler,
        "weight_decay": weight_decay,
        "max_duration": max_duration,
        "lora_r": lora_r,
        "lora_alpha": lora_alpha,
        "train_seed": train_seed,
        "inference_prompt": inference_prompt,
        "inference_duration": inference_duration,
        "inference_seed": inference_seed,
        "guidance_scale": guidance,
        "temperature": temp,
        "top_k": topk,
        "top_p": topp,
        "ollama_model": ollama_model
    }
    save_settings(settings_to_save)
    return "✅ Ajustes guardados en settings.json"

# === GRADIO ===
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎶 MusicGen DreamBooth – v4.8 (Parámetros Dinámicos)")
    initial_state = {"model": base_model, "lora_path": None}
    active_model_state = gr.State(value=initial_state)
    continuous_generation_state = gr.State(value={'is_running': False})

    # Timer declarado aquí (dentro de Blocks) - initially inactive
    timer = gr.Timer(10, active=False)

    with gr.Tabs():
        with gr.TabItem("🛠️ Entrenar LoRA"):
            save_settings_btn = gr.Button("💾 Guardar Todos los Ajustes")
            settings_save_output = gr.Textbox(label="Estado de los Ajustes", interactive=False)
            gr.Markdown("### 1. Preparación de Datos")
            prep_dataset_path_input = gr.Textbox(label="Ruta audios", value=settings.get("dataset_path", ""))
            generate_metadata_button = gr.Button("🤖 Generar metadata.jsonl")
            metadata_output = gr.Textbox(label="Resultado", lines=2)
            gr.Markdown("### Troceado de Dataset (para archivos largos)")
            chunked_output_path = gr.Textbox(label="Ruta salida troceado", value="./training_data")
            chunk_duration_input = gr.Slider(label="Duración de troceado (s)", minimum=5, maximum=60, value=30, step=1)
            chunk_dataset_btn = gr.Button("✂️ Trocear Dataset")
            gr.Markdown("### Aumentación de Dataset")
            augmented_output_path = gr.Textbox(label="Ruta salida augmentado", value="./augmented_training_data")
            use_augmented_cb = gr.Checkbox(label="Usar dataset augmentado", value=False)
            augment_dataset_btn = gr.Button("🔄 Augmentar Dataset")
            gr.Markdown("### Post-procesamiento de Descripciones")
            descriptions_dataset_path = gr.Textbox(label="Ruta dataset para mejorar descripciones", value="./training_data")
            use_ollama_for_descriptions = gr.Checkbox(label="Usar Ollama para mejorar descripciones", value=True)
            postprocess_descriptions_btn = gr.Button("📝 Mejorar Descripciones", variant="secondary")
            gr.Markdown("### 2. Parámetros de Entrenamiento")
            output_dir_input = gr.Textbox(label="Carpeta LoRA", value=settings.get("output_dir", ""))
            epochs_input = gr.Slider(label="Épocas", minimum=1, maximum=100, step=1, value=settings.get("epochs", 15))
            lr_input = gr.Number(label="LR", value=settings.get("lr", 0.0001), precision=6, step=0.00001)
            scheduler_input = gr.Dropdown(label="LR Scheduler", choices=["linear", "cosine", "constant"], value=settings.get("lr_scheduler", "linear"))
            weight_decay_input = gr.Slider(label="Weight Decay", minimum=0.0, maximum=0.2, step=0.01, value=settings.get("weight_decay", 0.01))
            max_duration_input = gr.Slider(label="Duración (s)", minimum=5, maximum=40, value=settings.get("max_duration", 8), step=1)
            r_input = gr.Slider(label="R", minimum=4, maximum=128, step=4, value=settings.get("lora_r", 8))
            alpha_input = gr.Slider(label="Alpha", minimum=4, maximum=256, step=4, value=settings.get("lora_alpha", 16))
            train_seed_input = gr.Number(label="Semilla", value=settings.get("train_seed", 42))
            gr.Markdown("### 3. Iniciar")
            launch_train_btn = gr.Button("🚀 Entrenar", variant="primary")
            interrupt_train_btn = gr.Button("🛑 Interrumpir")
            # Training log with custom CSS class for auto-scrolling
            train_log = gr.Textbox(label="Log", lines=15, interactive=False)

        with gr.TabItem("✍️ Gestor de Prompts"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_select_dd = gr.Dropdown(label="Prompts guardados", choices=prompt_manager.get_prompt_names())
                    prompt_name_tb = gr.Textbox(label="Nombre del Prompt")
                    save_prompt_btn = gr.Button("💾 Guardar")
                    delete_prompt_btn = gr.Button("🗑️ Eliminar")
                    prompt_status_tb = gr.Textbox(label="Estado", interactive=False)
                    prompt_text_area = gr.Textbox(label="Texto del Prompt", lines=5)
                with gr.Column(scale=1):
                    # Ollama Prompt Template Management
                    gr.Markdown("### 🤖 Plantillas Ollama")
                    ollama_template_select = gr.Dropdown(label="Plantillas Ollama", choices=ollama_prompt_manager.get_prompt_names())
                    ollama_template_name = gr.Textbox(label="Nombre Plantilla")
                    ollama_template_content = gr.Textbox(label="Contenido Plantilla", lines=4, placeholder="Use {prompt} como marcador para el prompt del usuario")
                    save_ollama_template_btn = gr.Button("💾 Guardar Plantilla")
                    delete_ollama_template_btn = gr.Button("🗑️ Eliminar Plantilla")
                    use_ollama_template_btn = gr.Button("🔄 Usar Plantilla", variant="primary")
            
            ollama_model_dd = gr.Dropdown(label="Modelo Ollama", choices=available_ollama_models, value=settings.get("ollama_model", ""))
            unload_ollama_btn = gr.Button("🗑️ Descargar modelo")
            free_gpu_btn = gr.Button("🧹 Liberar Memoria GPU")
            use_captions_cb = gr.Checkbox(label="Usar captions del dataset como contexto")
            enhance_btn = gr.Button("🔧 Mejorar con Ollama", variant="primary")
            use_in_inference_btn = gr.Button("🎵 Usar en Generador")

        with gr.TabItem("🎵 Generador"):
            with gr.Row():
                prompt_input = gr.Textbox(label="Prompt", value=settings.get("inference_prompt", "tonetxo_style, synthwave"), show_copy_button=True)
                lora_path_input = gr.File(label="Arrastra LoRA", file_count="multiple")
            with gr.Row():
                inference_seed_input = gr.Number(label="Semilla", value=settings.get("inference_seed", -1))
                duration_slider = gr.Slider(label="Duración (s)", minimum=5, maximum=40, value=settings.get("inference_duration", 8), step=1)
            with gr.Row():
                generate_btn = gr.Button("🎹 Generar", variant="primary")
                continuous_generate_btn = gr.Button("🔄 Generación Continua", variant="secondary")
                stop_generate_btn = gr.Button("🛑 Detener", variant="stop")
            
            # 👇 Campo de intervalo AÑADIDO aquí
            interval_input = gr.Number(label="Intervalo entre generaciones (segundos)", value=10, minimum=1, step=1)

            status_output = gr.Textbox(label="Estado", value="✅ Modelo Base")
            with gr.Accordion("🎛️ Parámetros de Generación (Modificables en tiempo real)", open=True):
                with gr.Row():
                    guidance_slider = gr.Slider(label="CFG Scale", minimum=1, maximum=20, value=settings.get("guidance_scale", 3.0), step=0.1)
                    temperature_slider = gr.Slider(label="Temperature", minimum=0.1, maximum=2.0, value=settings.get("temperature", 1.0), step=0.1)
                with gr.Row():
                    topk_slider = gr.Slider(label="Top-k", minimum=0, maximum=500, value=settings.get("top_k", 250), step=1)
                    topp_slider = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=settings.get("top_p", 0.0), step=0.01)
            audio_out = gr.Audio(label="Audio Generado", type="numpy")
            with gr.Row():
                save_audio_btn = gr.Button("💾 Guardar Audio")
                save_output = gr.Textbox(label="Estado Guardado", interactive=False)

    # --- Eventos ---
    all_settings_comps = [
        prep_dataset_path_input, output_dir_input, epochs_input, lr_input, scheduler_input, 
        weight_decay_input, max_duration_input, r_input, alpha_input, train_seed_input,
        prompt_input, duration_slider, inference_seed_input, guidance_slider, 
        temperature_slider, topk_slider, topp_slider, ollama_model_dd
    ]
    save_settings_btn.click(save_all_settings, inputs=all_settings_comps, outputs=settings_save_output)
    generate_metadata_button.click(generate_metadata, inputs=prep_dataset_path_input, outputs=metadata_output)
    chunk_dataset_btn.click(chunk_dataset, inputs=[prep_dataset_path_input, chunked_output_path, chunk_duration_input], outputs=metadata_output)
    augment_dataset_btn.click(augment_dataset_simple, inputs=[prep_dataset_path_input, augmented_output_path], outputs=metadata_output)
    postprocess_descriptions_btn.click(postprocess_descriptions, inputs=[descriptions_dataset_path, use_ollama_for_descriptions, ollama_model_dd], outputs=metadata_output)
    launch_train_btn.click(
        modify_and_run_training,
        inputs=[
            prep_dataset_path_input, output_dir_input, epochs_input, lr_input, scheduler_input, 
            r_input, alpha_input, max_duration_input, train_seed_input, use_augmented_cb, 
            augmented_output_path, weight_decay_input
        ],
        outputs=train_log
    )
    interrupt_train_btn.click(interrupt_training, outputs=train_log)
    prompt_select_dd.change(lambda name: prompt_manager.get_prompt_text(name), inputs=prompt_select_dd, outputs=prompt_text_area)
    save_prompt_btn.click(lambda name, text: (prompt_manager.update_prompt(name, text), gr.Dropdown(choices=prompt_manager.get_prompt_names(), value=name)), inputs=[prompt_name_tb, prompt_text_area], outputs=[prompt_status_tb, prompt_select_dd])
    delete_prompt_btn.click(lambda name: (prompt_manager.delete_prompt(name), gr.Dropdown(choices=prompt_manager.get_prompt_names())), inputs=prompt_name_tb, outputs=[prompt_status_tb, prompt_select_dd])
    # Enhanced version that uses selected Ollama template
    enhance_btn.click(
        lambda model, prompt, use_captions, dataset_path, template_name: 
            ollama.enhance_and_translate_prompt(
                model, prompt, use_captions, dataset_path,
                ollama_prompt_manager.get_prompt_template(template_name) if template_name else None
            ),
        inputs=[ollama_model_dd, prompt_text_area, use_captions_cb, prep_dataset_path_input, ollama_template_select],
        outputs=prompt_text_area
    )
    unload_ollama_btn.click(ollama.unload_ollama_model, inputs=[ollama_model_dd], outputs=prompt_status_tb)
    free_gpu_btn.click(free_gpu_memory, outputs=prompt_status_tb)
    use_in_inference_btn.click(lambda txt: txt, inputs=prompt_text_area, outputs=prompt_input)
    lora_path_input.change(switch_model_and_state, inputs=lora_path_input, outputs=[active_model_state, status_output])
    
    # --- Eventos para Plantillas Ollama ---
    ollama_template_select.change(
        lambda name: ollama_prompt_manager.get_prompt_template(name), 
        inputs=ollama_template_select, 
        outputs=ollama_template_content
    )
    save_ollama_template_btn.click(
        lambda name, template: (
            ollama_prompt_manager.update_prompt(name, template), 
            gr.Dropdown(choices=ollama_prompt_manager.get_prompt_names(), value=name)
        ), 
        inputs=[ollama_template_name, ollama_template_content], 
        outputs=[prompt_status_tb, ollama_template_select]
    )
    delete_ollama_template_btn.click(
        lambda name: (
            ollama_prompt_manager.delete_prompt(name), 
            gr.Dropdown(choices=ollama_prompt_manager.get_prompt_names())
        ), 
        inputs=ollama_template_name, 
        outputs=[prompt_status_tb, ollama_template_select]
    )
    use_ollama_template_btn.click(
        lambda name: ollama_prompt_manager.get_prompt_template(name), 
        inputs=ollama_template_select, 
        outputs=ollama_template_content
    )

    # --- Eventos de Generación ---
    # Generación única
    generate_btn.click(
        generate_music_with_state,
        inputs=[active_model_state, lora_path_input, prompt_input, duration_slider, inference_seed_input, guidance_slider, temperature_slider, topk_slider, topp_slider],
        outputs=[active_model_state, status_output, audio_out]
    ).then(
        lambda audio_data: save_generated_audio(audio_data) if audio_data else "❌ No hay audio para guardar",
        inputs=[audio_out],
        outputs=[save_output]
    )

    # === NUEVO: GENERACIÓN CONTINUA CON TIMER ===
    def start_continuous_timer(state):
        state["is_running"] = True
        return state, gr.Timer(active=True), "🔄 Generación continua iniciada..."

    def stop_continuous_timer(state):
        state["is_running"] = False
        return state, gr.Timer(active=False), "🛑 Generación continua detenida"

    def timer_step(
        continuous_state,
        model_state,
        lora_files,
        prompt,
        duration,
        seed,
        guidance,
        temp,
        topk,
        topp,
    ):
        if not continuous_state.get("is_running", False):
            return continuous_state, model_state, "⏹️ Pausado", None
        try:
            new_state, status, audio = generate_music_with_state(
                model_state, lora_files, prompt, duration, seed, guidance, temp, topk, topp
            )
            if audio is not None:
                save_result = save_generated_audio(audio)
                status = f"🔄 Generado | {save_result}"
            return continuous_state, new_state, status, audio
        except Exception as e:
            logger.error(f"Error en generación continua: {e}")
            return continuous_state, model_state, f"❌ Error: {str(e)}", None

    # Actualizar el timer cuando cambie el intervalo
    interval_input.change(
        lambda x: gr.Timer(active=False) if x <= 0 else gr.Timer(value=x, active=False),
        inputs=interval_input,
        outputs=timer
    )

    continuous_generate_btn.click(
        start_continuous_timer,
        inputs=[continuous_generation_state],
        outputs=[continuous_generation_state, timer, status_output]
    )

    stop_generate_btn.click(
        stop_continuous_timer,
        inputs=[continuous_generation_state],
        outputs=[continuous_generation_state, timer, status_output]
    )

    timer.tick(
        timer_step,
        inputs=[
            continuous_generation_state,
            active_model_state,
            lora_path_input,
            prompt_input,
            duration_slider,
            inference_seed_input,
            guidance_slider,
            temperature_slider,
            topk_slider,
            topp_slider,
        ],
        outputs=[
            continuous_generation_state,
            active_model_state,
            status_output,
            audio_out,
        ]
    )

    save_audio_btn.click(
        save_generated_audio,
        inputs=[audio_out],
        outputs=[save_output]
    )

# Custom JavaScript to auto-scroll training log to bottom
auto_scroll_js = """
function() {
    // Find the training log textbox by its label
    var logElement = document.querySelector('textarea[aria-label="Log"]') || 
                     document.querySelector('textarea[placeholder="Log"]') ||
                     document.querySelector('#component-0 textarea'); // fallback to first textarea
    if (logElement) {
        logElement.scrollTop = logElement.scrollHeight;
    }
    return [];
}
"""

if __name__ == "__main__":
    demo.launch()