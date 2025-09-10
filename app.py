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

import gradio as gr
import librosa
import requests
import torch
import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

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
def _extract_description(raw: Any) -> str:
    """
    Convierte la salida del tagger en una cadena que se usará como campo
    `description` del caption completo.
    Se buscan los mismos campos que antes (caption, label, …) y se devuelve
    una cadena legible. Si no se encuentra nada devuelve un placeholder.
    """
    if isinstance(raw, dict):
        for key in (
            "caption",
            "text",
            "label",
            "labels",
            "tags",
            "predictions",
            "prediction",
            "description",
        ):
            if key in raw and raw[key]:
                val = raw[key]
                if isinstance(val, (list, tuple)):
                    return ", ".join([str(v).strip() for v in val if v])
                return str(val).strip()
    if isinstance(raw, (list, tuple)):
        return ", ".join([str(v).strip() for v in raw if v])
    if isinstance(raw, str):
        return raw.strip()
    for attr in ("caption", "text"):
        if hasattr(raw, attr):
            val = getattr(raw, attr)
            if val:
                return str(val).strip()
    return "audio caption placeholder"


# --------------------------------------------------------------------------- #
# FUNCIÓN DE METADATA ------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def generate_metadata(dataset_dir: str) -> str:
    """
    Recorre `dataset_dir` y crea un `metadata.jsonl` con el **formato completo**
    que indicas.
    """
    if not tagger_loaded:
        return "❌ El modelo de audio‑tagging no está disponible. Revisa los logs."

    root = Path(dataset_dir)
    if not root.is_dir():
        return f"❌ Ruta de dataset inválida: {dataset_dir}"

    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    if not audio_files:
        return f"⚠️ No se encontraron archivos de audio en {dataset_dir}"

    out_path = root / "metadata.jsonl"
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
                description = "audio caption placeholder"
            else:
                description = _extract_description(raw_res)

            caption_dict = {
                "key": "", "artist": "Voyager I", "sample_rate": int(sr),
                "file_extension": audio_path.suffix.lstrip("."),
                "description": description, "keywords": "", "duration": round(duration, 2),
                "bpm": "", "genre": "electronic", "title": "Untitled song",
                "name": audio_path.stem, "instrument": "Mix", "moods": [],
            }
            out_f.write(json.dumps(caption_dict, ensure_ascii=False) + "\n")

    return f"✅ metadata.jsonl creado en: {out_path}"


# --------------------------------------------------------------------------- #
# ENTRENAMIENTO (DreamBooth) ------------------------------------------------ #
# --------------------------------------------------------------------------- #
def modify_and_run_training(
    dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, train_seed
) -> Generator[str, None, None]:
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
        yield f"❌ Error al modificar script: {e}"
        return

    command = [
        "accelerate", "launch", "dreambooth_musicgen.py",
        f"--model_name_or_path={MODEL_ID}", f"--dataset_name={dataset_path}",
        f"--output_dir={output_dir}", f"--num_train_epochs={int(epochs)}",
        "--use_lora", f"--learning_rate={lr}", "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=4", "--fp16", "--text_column_name=description",
        "--target_audio_column_name=audio_filepath", "--train_split_name=train",
        "--overwrite_output_dir", "--do_train", "--decoder_start_token_id=2048",
        f"--max_duration_in_seconds={int(max_duration)}", "--gradient_checkpointing",
        f"--seed={int(train_seed)}",
    ]

    yield "🚀 Lanzando entrenamiento...\n\n"
    process = subprocess.Popen(
        command, cwd="./musicgen-dreamboothing", stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, encoding="utf-8",
    )
    for line in iter(process.stdout.readline, ""):
        yield line
    process.wait()
    if process.returncode == 0:
        yield "\n✅ ¡Entrenamiento finalizado!"
    else:
        yield f"\n❌ Proceso terminó con código {process.returncode}"


# --------------------------------------------------------------------------- #
# INFERENCIA – Gestión de LoRA ---------------------------------------------- #
# --------------------------------------------------------------------------- #
def switch_model_and_state(lora_files: List[str]) -> Tuple[Dict[str, Any], str]:
    """
    Carga un LoRA a partir de una lista de archivos temporales de Gradio.
    """
    logger.info(f"Cambiando modelo. Archivos LoRA recibidos: {lora_files}")

    # Usar la variable global base_model
    global base_model
    
    # Liberar memoria primero
    if hasattr(base_model, 'to'):
        base_model.to("cpu")
    torch.cuda.empty_cache()

    # Si la lista está vacía o es None, volvemos al modelo base ORIGINAL
    if not lora_files or all(f is None for f in lora_files):
        try:
            # RECREAR el modelo base desde cero para asegurar que no hay LoRA
            logger.info("Recreando modelo base limpio...")
            
            # Liberar memoria del modelo actual
            del base_model
            torch.cuda.empty_cache()
            
            # Recargar el modelo base completamente desde cero
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            
            # Reaplicar configuración de flash attention
            if hasattr(base_model.config, "use_flash_attention_2"):
                base_model.config.use_flash_attention_2 = True
            if hasattr(base_model, "text_encoder"):
                if hasattr(base_model.text_encoder.config, "use_flash_attention_2"):
                    base_model.text_encoder.config.use_flash_attention_2 = False
            
            logger.info("✅ Modelo base recreado exitosamente")
            
        except Exception as e:
            logger.error(f"Error recreando modelo base: {e}")
            # Si falla, intentar cargar de nuevo el modelo
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
        
        active_model = base_model
        status = "✅ Modelo Base Activo (limpio)"
        new_state = {"model": base_model, "lora_path": None}
        
    else:
        try:
            # Crear un directorio temporal único para los archivos LoRA
            import tempfile
            import shutil
            
            lora_temp_dir = tempfile.mkdtemp()
            logger.info(f"Creando directorio temporal para LoRA: {lora_temp_dir}")
            
            # Copiar todos los archivos al directorio temporal
            for file_path in lora_files:
                if file_path and file_path != "None":
                    filename = os.path.basename(file_path)
                    dest_path = os.path.join(lora_temp_dir, filename)
                    shutil.copy2(file_path, dest_path)
                    logger.info(f"Copiado {filename} a {lora_temp_dir}")
            
            # Verificar que tenemos los archivos necesarios
            files_in_dir = os.listdir(lora_temp_dir)
            logger.info(f"Archivos en directorio temporal: {files_in_dir}")
            
            # Asegurarse de que el modelo base esté limpio primero
            if hasattr(base_model, 'peft_config'):
                logger.info("Limpiando modelo base antes de cargar LoRA...")
                del base_model
                torch.cuda.empty_cache()
                base_model = MusicgenForConditionalGeneration.from_pretrained(
                    MODEL_ID,
                    torch_dtype=torch.float16,
                )
            
            # Cargar el LoRA desde el directorio temporal
            active_model = PeftModel.from_pretrained(base_model, lora_temp_dir)

            if hasattr(active_model.config, "use_flash_attention_2"):
                active_model.config.use_flash_attention_2 = True
            if hasattr(active_model, "text_encoder"):
                if hasattr(active_model.text_encoder.config, "use_flash_attention_2"):
                    active_model.text_encoder.config.use_flash_attention_2 = False

            status = f"✅ LoRA activo"
            new_state = {"model": active_model, "lora_path": lora_temp_dir}
            
        except Exception as e:
            logger.error(f"Error cargando LoRA: {e}")
            # Limpiar el directorio temporal en caso de error
            if 'lora_temp_dir' in locals():
                shutil.rmtree(lora_temp_dir, ignore_errors=True)
            
            # Asegurarse de tener un modelo base limpio
            base_model = MusicgenForConditionalGeneration.from_pretrained(
                MODEL_ID,
                torch_dtype=torch.float16,
            )
            active_model = base_model
            status = f"❌ Error al cargar LoRA: {e}"
            new_state = {"model": base_model, "lora_path": None}
    
    active_model.to("cpu")
    torch.cuda.empty_cache()
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
    # Detectar si queremos volver al modelo base
    want_base_model = (not lora_path_from_file_input or 
                      all(f is None for f in lora_path_from_file_input))
    
    # Detectar si actualmente tenemos LoRA cargado
    has_lora_loaded = current_state["lora_path"] is not None
    
    # Siempre forzar el cambio si queremos modelo base pero tenemos LoRA
    if want_base_model and has_lora_loaded:
        logger.info("Forzando cambio a modelo base...")
        new_state, status = switch_model_and_state([])
        active_model = new_state["model"]
        current_state = new_state
    elif not want_base_model and not has_lora_loaded:
        # Queremos LoRA pero tenemos modelo base
        new_state, status = switch_model_and_state(lora_path_from_file_input)
        active_model = new_state["model"]
        current_state = new_state
    elif not want_base_model and has_lora_loaded:
        # Queremos LoRA y ya tenemos uno cargado, verificar si son diferentes
        current_files = set()
        if current_state["lora_path"] and os.path.exists(current_state["lora_path"]):
            current_files = set(os.listdir(current_state["lora_path"]))
        
        new_files = set()
        for file_path in lora_path_from_file_input:
            if file_path and file_path != "None":
                new_files.add(os.path.basename(file_path))
        
        # Si los archivos son diferentes, necesitamos cambiar
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

    active_model.to("cuda")
    logger.info("Generando audio en GPU...")

    if seed is not None and int(seed) != -1:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))

    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    if topk == 0 and topp == 0.0:
        topk = 250

    audio = active_model.generate(
        **inputs, max_new_tokens=int(duration * 50), do_sample=True,
        guidance_scale=guidance, temperature=temp, top_k=int(topk),
        top_p=topp if topp > 0 else None,
    )

    active_model.to("cpu")
    torch.cuda.empty_cache()
    logger.info("✅ Generación completada, VRAM liberada")

    sr = base_model.config.audio_encoder.sampling_rate
    audio_np = audio.cpu().numpy().squeeze()
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
                metadata_output = gr.Textbox(
                    label="Resultado", lines=2, interactive=False
                )
            with gr.Row():
                output_dir_input = gr.Textbox(
                    label="📁 Carpeta de salida (LoRA)",
                    value=settings.get("output_dir", ""),
                )
                epochs_input = gr.Slider(
                    label="Épocas", minimum=1, maximum=100, step=1, value=settings.get("epochs", 15)
                )
                lr_input = gr.Number(label="Learning Rate", value=settings.get("lr", 0.0001))
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
    launch_train_btn.click(fn=modify_and_run_training, inputs=[prep_dataset_path_input, output_dir_input, epochs_input, lr_input, r_input, alpha_input, max_duration_input, train_seed_input], outputs=train_log)

    lora_path_input.change(fn=switch_model_and_state, inputs=lora_path_input, outputs=[active_model_state, status_output])

    generate_btn.click(fn=generate_music_with_state,
        inputs=[
            active_model_state, lora_path_input, prompt_input, duration_slider,
            inference_seed_input, guidance_slider, temperature_slider, topk_slider, topp_slider,
        ],
        outputs=[active_model_state, status_output, audio_out]
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