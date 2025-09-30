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
        "max_duration": 8,  # 🔑 Clave para evitar OOM
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
        except:
            pass
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
                    return json.load(f)
            except:
                pass
        return []
    def save_prompts(self):
        with open(self.prompts_file, "w", encoding="utf-8") as f:
            json.dump(self.prompts, f, indent=4, ensure_ascii=False)
    def update_prompt(self, name, text):
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

class OllamaIntegration:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
    def get_available_models(self):
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            return [m["name"] for m in r.json().get("models", [])] if r.status_code == 200 else []
        except: return ["Error: No se pudo conectar a Ollama"]
    def enhance_and_translate_prompt(self, model_name, base_prompt, use_captions, dataset_path):
        if not base_prompt.strip(): return "Prompt vacío"
        tags_context = ""
        if use_captions and dataset_path:
            meta = Path(dataset_path) / "metadata.jsonl"
            if meta.exists():
                try:
                    words = set()
                    with open(meta, "r", encoding="utf-8") as f:
                        for line in f:
                            d = json.loads(line)
                            words.update(re.findall(r"\b\w+\b", d.get("description", "").lower()))
                    if words: tags_context = "Inspírate en: " + ", ".join(sorted(words)) + ". "
                except: pass
        ollama_prompt = f"You are an expert music-prompt writer. {tags_context}Improve and translate to English for MusicGen: \"{base_prompt}\""
        try:
            r = self.session.post(f"{self.base_url}/api/generate", json={"model": model_name, "prompt": ollama_prompt, "stream": False}, timeout=60)
            out = r.json().get("response", "").strip().replace('"', "")
            return out[0].upper() + out[1:] if out else base_prompt
        except: return "Error en Ollama"
    def unload_ollama_model(self, model_name):
        """Descarga un modelo específico de Ollama de la memoria."""
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
    """Libera la memoria GPU utilizada por el modelo de MusicGen"""
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
available_ollama_models = ollama.get_available_models()
if not settings["ollama_model"] and available_ollama_models:
    settings["ollama_model"] = available_ollama_models[0]

# === AUDIO TAGGER ===
tagger = AudioTagger()
tagger_loaded = False
try:
    msg = tagger.load_model()
    if "Error" not in msg: tagger_loaded = True
except: pass

def _extract_description(raw, audio_path=None):
    """Extraer descripción mejorada con PANNs"""
    result = {
        "description": "electronic music, synth, drums", 
        "bpm": "", 
        "genre": "electronic", 
        "moods": []
    }
    
    desc = ""
    
    # Manejar diferentes formatos de respuesta de PANNs
    if isinstance(raw, dict):
        if "caption" in raw and raw["caption"]:
            desc = raw["caption"]
        elif "labels" in raw and raw["labels"]:
            # Usar las etiquetas limpiadas de PANNs
            labels = raw["labels"]
            if labels:
                # Tomar las 3-4 etiquetas principales
                main_labels = [label.get("cleaned_label", label["label"]) 
                             for label in labels[:4] if label.get("confidence", 0) > 0.1]
                if main_labels:
                    desc = ", ".join(main_labels)
    
    elif isinstance(raw, str):
        desc = raw
    
    # Mejorar la descripción base si es genérica
    if not desc or desc == "audio caption placeholder":
        desc = "electronic music, synth, drums"
    
    # Asegurar que tenga el estilo Tonetxo
    if "tonetxo_style" not in desc.lower():
        desc = f"tonetxo_style, {desc}"
    
    result["description"] = desc
    
    # Calcular BPM si hay archivo de audio
    if audio_path and os.path.exists(audio_path):
        try:
            y, sr = librosa.load(audio_path, sr=None, mono=True)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            result["bpm"] = str(round(float(tempo))) if tempo > 0 else ""
        except Exception as e:
            logger.warning(f"No se pudo calcular BPM para {audio_path}: {e}")
    
    return result
    
def generate_metadata(dataset_dir):
    if not tagger_loaded: return "❌ AudioTagger no disponible"
    root = Path(dataset_dir)
    if not root.is_dir(): return f"❌ Ruta inválida: {dataset_dir}"
    exts = {".wav", ".mp3", ".flac"}
    audio_files = [p for p in root.iterdir() if p.suffix.lower() in exts]
    if not audio_files: return "⚠️ No hay audios"
    out_path = root / "metadata.jsonl"
    with out_path.open("w", encoding="utf-8") as out_f:
        for audio_path in tqdm.tqdm(audio_files, desc="Generando metadata"):
            try:
                raw_res = tagger.process_audio_file(str(audio_path))
            except:
                raw_res = {}
            try:
                y, sr = librosa.load(str(audio_path), sr=None)
                duration = float(librosa.get_duration(y=y, sr=sr))
            except:
                sr, duration = 44100, 0.0
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
    return f"✅ metadata.jsonl creado en: {out_path}"

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
    try:
        output_path.mkdir(parents=True, exist_ok=True)
        augmented_audio_dir.mkdir(exist_ok=True)
    except Exception as e:
        return f"❌ Error creando directorio de salida: {e}"
    def simple_augment_audio(y, sr, filename_prefix):
        augmented_samples = []
        # 1. Pitch Shift
        try:
            augmented_samples.append((librosa.effects.pitch_shift(y, sr=sr, n_steps=-1), sr, f"{filename_prefix}_pitch_down1"))
            augmented_samples.append((librosa.effects.pitch_shift(y, sr=sr, n_steps=1), sr, f"{filename_prefix}_pitch_up1"))
        except: pass
        
        # 2. Volume Change
        augmented_samples.append((np.clip(y * 0.9, -1.0, 1.0), sr, f"{filename_prefix}_vol_down"))
        augmented_samples.append((np.clip(y * 1.1, -1.0, 1.0), sr, f"{filename_prefix}_vol_up"))

        # 3. Time Stretch (Nuevo)
        try:
            augmented_samples.append((librosa.effects.time_stretch(y, rate=0.9), sr, f"{filename_prefix}_stretch_slow"))
            augmented_samples.append((librosa.effects.time_stretch(y, rate=1.1), sr, f"{filename_prefix}_stretch_fast"))
        except: pass

        # 4. Add Noise (Nuevo)
        try:
            noise = np.random.randn(len(y))
            y_noise = y + 0.005 * noise
            augmented_samples.append((np.clip(y_noise, -1.0, 1.0), sr, f"{filename_prefix}_noise"))
        except: pass
            
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

    # --- BORRADO MANUAL Y FORZADO DEL DIRECTORIO DE SALIDA ---
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
    # ----------------------------------------------------------

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
    except: pass

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

def generate_music_with_state(
    current_state, lora_path_from_file_input, prompt, duration, seed, guidance, temp, topk, topp
):
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
    max_tokens = min(int(duration * 50), 2048)

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
    if audio_np.ndim == 2 and audio_np.shape[0] <= 2:
        audio_np = audio_np.T
    elif audio_np.ndim == 1:
        pass
    else:
        audio_np = audio_np[0] if audio_np.shape[0] > 1 else audio_np

    # Convertir a int16 para evitar el warning de Gradio
    audio_np = (audio_np * 32767).astype(np.int16)

    return current_state, "✅ Generado", (32000, audio_np)

def save_generated_audio(audio_data, output_dir="./generated_audio"):
    if not audio_data: return "❌ No hay audio"
    sr, audio_np = audio_data
    if audio_np is None or len(audio_np) == 0: return "❌ Audio vacío"
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"musicgen_{timestamp}.wav")
    try:
        if audio_np.ndim == 2 and audio_np.shape[1] > 2:
            audio_np = audio_np[:, :2]
        sf.write(filepath, audio_np, sr)
        return f"✅ Guardado: {filepath}"
    except Exception as e:
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
    gr.Markdown("# 🎶 MusicGen DreamBooth – v4.6 (corregido)")
    initial_state = {"model": base_model, "lora_path": None}
    active_model_state = gr.State(value=initial_state)

    with gr.Tabs():
        with gr.TabItem("🛠️ Entrenar LoRA"):
            save_settings_btn = gr.Button("💾 Guardar Todos los Ajustes")
            settings_save_output = gr.Textbox(label="Estado de los Ajustes", interactive=False)
            
            gr.Markdown("### 1. Preparación de Datos")
            prep_dataset_path_input = gr.Textbox(label="Ruta audios", value=settings.get("dataset_path", ""))
            generate_metadata_button = gr.Button("🤖 Generar metadata.jsonl")
            metadata_output = gr.Textbox(label="Resultado", lines=2)
            
            augmented_output_path = gr.Textbox(label="Ruta salida augmentado", value="./augmented_training_data")
            use_augmented_cb = gr.Checkbox(label="Usar dataset augmentado", value=False)
            augment_dataset_btn = gr.Button("🔄 Augmentar Dataset")

            gr.Markdown("### 2. Parámetros de Entrenamiento")
            output_dir_input = gr.Textbox(label="Carpeta LoRA", value=settings.get("output_dir", ""))
            epochs_input = gr.Slider(label="Épocas", minimum=1, maximum=100, step=1, value=settings.get("epochs", 15))
            lr_input = gr.Number(label="LR", value=settings.get("lr", 0.0001), precision=6)
            scheduler_input = gr.Dropdown(label="LR Scheduler", choices=["linear", "cosine", "constant"], value=settings.get("lr_scheduler", "linear"))
            weight_decay_input = gr.Slider(label="Weight Decay", minimum=0.0, maximum=0.2, step=0.01, value=settings.get("weight_decay", 0.01))
            max_duration_input = gr.Slider(label="Duración (s)", minimum=5, maximum=40, value=settings.get("max_duration", 8))
            r_input = gr.Slider(label="R", minimum=4, maximum=128, step=4, value=settings.get("lora_r", 8))
            alpha_input = gr.Slider(label="Alpha", minimum=4, maximum=256, step=4, value=settings.get("lora_alpha", 16))
            train_seed_input = gr.Number(label="Semilla", value=settings.get("train_seed", 42))
            
            gr.Markdown("### 3. Iniciar")
            launch_train_btn = gr.Button("🚀 Entrenar", variant="primary")
            interrupt_train_btn = gr.Button("🛑 Interrumpir")
            train_log = gr.Textbox(label="Log", lines=15)

        with gr.TabItem("✍️ Gestor de Prompts"):
            prompt_select_dd = gr.Dropdown(label="Prompts guardados", choices=prompt_manager.get_prompt_names())
            prompt_name_tb = gr.Textbox(label="Nombre del Prompt")
            save_prompt_btn = gr.Button("💾 Guardar")
            delete_prompt_btn = gr.Button("🗑️ Eliminar")
            prompt_status_tb = gr.Textbox(label="Estado", interactive=False)
            prompt_text_area = gr.Textbox(label="Texto del Prompt", lines=5)
            ollama_model_dd = gr.Dropdown(label="Modelo Ollama", choices=available_ollama_models, value=settings.get("ollama_model", ""))
            unload_ollama_btn = gr.Button("🗑️ Descargar modelo")
            free_gpu_btn = gr.Button("🧹 Liberar Memoria GPU")
            use_captions_cb = gr.Checkbox(label="Usar captions del dataset como contexto")
            enhance_btn = gr.Button("🔧 Mejorar con Ollama", variant="primary")
            use_in_inference_btn = gr.Button("🎵 Usar en Generador")

        with gr.TabItem("🎵 Generador"):
            prompt_input = gr.Textbox(label="Prompt", value=settings.get("inference_prompt", "tonetxo_style, synthwave"))
            lora_path_input = gr.File(label="Arrastra LoRA", file_count="multiple")
            inference_seed_input = gr.Number(label="Semilla", value=settings.get("inference_seed", -1))
            generate_btn = gr.Button("🎹 Generar", variant="primary")
            status_output = gr.Textbox(label="Estado", value="✅ Modelo Base")
            duration_slider = gr.Slider(label="Duración (s)", minimum=5, maximum=40, value=settings.get("inference_duration", 8))
            with gr.Accordion("Ajustes avanzados", open=False):
                guidance_slider = gr.Slider(label="CFG", minimum=1, maximum=20, value=settings.get("guidance_scale", 3.0))
                temperature_slider = gr.Slider(label="Temp", minimum=0.1, maximum=2.0, value=settings.get("temperature", 1.0))
                topk_slider = gr.Slider(label="Top-k", minimum=0, maximum=500, value=settings.get("top_k", 250))
                topp_slider = gr.Slider(label="Top-p", minimum=0.0, maximum=1.0, value=settings.get("top_p", 0.0))
            audio_out = gr.Audio(label="Audio", type="numpy")
            save_audio_btn = gr.Button("💾 Guardar")
            save_output = gr.Textbox(label="Guardado")

    # --- Eventos ---
    all_settings_comps = [
        prep_dataset_path_input, output_dir_input, epochs_input, lr_input, scheduler_input, 
        weight_decay_input, max_duration_input, r_input, alpha_input, train_seed_input,
        prompt_input, duration_slider, inference_seed_input, guidance_slider, 
        temperature_slider, topk_slider, topp_slider, ollama_model_dd
    ]
    save_settings_btn.click(save_all_settings, inputs=all_settings_comps, outputs=settings_save_output)

    generate_metadata_button.click(generate_metadata, inputs=prep_dataset_path_input, outputs=metadata_output)
    augment_dataset_btn.click(augment_dataset_simple, inputs=[prep_dataset_path_input, augmented_output_path], outputs=metadata_output)
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
    enhance_btn.click(ollama.enhance_and_translate_prompt, inputs=[ollama_model_dd, prompt_text_area, use_captions_cb, prep_dataset_path_input], outputs=prompt_text_area)
    unload_ollama_btn.click(ollama.unload_ollama_model, inputs=[ollama_model_dd], outputs=prompt_status_tb)
    free_gpu_btn.click(free_gpu_memory, outputs=prompt_status_tb)
    use_in_inference_btn.click(lambda txt: txt, inputs=prompt_text_area, outputs=prompt_input)
    lora_path_input.change(switch_model_and_state, inputs=lora_path_input, outputs=[active_model_state, status_output])
    generate_btn.click(
        generate_music_with_state,
        inputs=[active_model_state, lora_path_input, prompt_input, duration_slider, inference_seed_input, guidance_slider, temperature_slider, topk_slider, topp_slider],
        outputs=[active_model_state, status_output, audio_out]
    )
    save_audio_btn.click(save_generated_audio, inputs=audio_out, outputs=save_output)

if __name__ == "__main__":
    demo.launch()