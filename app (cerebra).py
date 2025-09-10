
# -*- coding: utf-8 -*-
# app.py (versión 4.3 – Integración con captioning.py, Flash‑Attention y corrección de captions)

"""
Flujo completo:
1️⃣  Gestor de prompts (guardar / cargar / eliminar)
2️⃣  Mejora de prompts con Ollama (opcionalmente usando keywords extraídas del
    metadata.jsonl generado a partir del dataset).
3️⃣  Generación de `metadata.jsonl` con un modelo de *audio‑tagging* real
    (ahora desde captioning.py).  Cada línea contiene:
        {"text": "<caption generada>", "audio_filepath": "..."}
4️⃣  Entrenamiento DreamBooth (script externo) → LoRA.
5️⃣  Inferencia: selección dinámica del LoRA + generación de audio con MusicGen
    (con Flash‑Attention activado, **excepto** el encoder del T5).
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
import requests
import torch
import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

# --------------------------------------------------------------------------- #
# IMPORT DE AUDIO‑TAGGER ------------------------------------------------------ #
# --------------------------------------------------------------------------- #
# El nuevo tagger está definido en `captioning.py`. Sólo lo importamos aquí.
from captioning import AudioTagger

# --------------------------------------------------------------------------- #
# LOGGING ------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
logger = logging.getLogger("MusicGenApp")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
logger.addHandler(handler)

# --------------------------------------------------------------------------- #
# CONFIGURACIÓN GLOBAL ------------------------------------------------------ #
# --------------------------------------------------------------------------- #
SETTINGS_FILE = Path("settings.json")
MODEL_ID = "facebook/musicgen-small"


def load_settings() -> Dict[str, Any]:
    """Carga o crea `settings.json` con valores por defecto."""
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
    if SETTINGS_FILE.is_file():
        try:
            with SETTINGS_FILE.open("r", encoding="utf-8") as f:
                overrides = json.load(f)
                defaults.update(overrides)
        except (json.JSONDecodeError, IOError):
            logger.warning("No se pudo leer settings.json – se usan valores por defecto")
    return defaults


def save_settings(settings: Dict[str, Any]) -> None:
    """Persistencia de configuración en disco."""
    try:
        with SETTINGS_FILE.open("w", encoding="utf-8") as f:
            json.dump(settings, f, indent=4, ensure_ascii=False)
        logger.info("✅ Settings guardados")
    except IOError as e:
        logger.error(f"Error guardando settings: {e}")


settings = load_settings()

# --------------------------------------------------------------------------- #
# CARGA DEL MODELO BASE (CPU) + PROCESSOR ----------------------------------- #
# --------------------------------------------------------------------------- #
logger.info("Cargando modelo base y procesador (CPU)…")

# --------------------------------------------------------------------------- #
# FLASH‑ATTENTION ----------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Habilitamos Flash‑Attention 2 en todo el modelo **excepto** el encoder del T5,
# que todavía no es compatible con esta optimización.
processor = AutoProcessor.from_pretrained(MODEL_ID)

base_model = MusicgenForConditionalGeneration.from_pretrained(
    MODEL_ID,
    dtype=torch.float16,                     # ← argumento correcto
)
# Activamos Flash‑Attention en el modelo (si la opción está disponible)
if hasattr(base_model.config, "use_flash_attention_2"):
    base_model.config.use_flash_attention_2 = True
    logger.info("⚡ Flash‑Attention 2 activado en MusicGen.")
# Desactivamos la flag en el encoder de texto (T5) para evitar incompatibilidades.
if hasattr(base_model, "text_encoder"):
    if hasattr(base_model.text_encoder.config, "use_flash_attention_2"):
        base_model.text_encoder.config.use_flash_attention_2 = False
        logger.info(
            "⚡ Flash‑Attention 2 **desactivado** en el encoder del texto (T5)."
        )
base_model.to("cpu")
logger.info("✔️ Modelo base listo")

# --------------------------------------------------------------------------- #
# CLASE DE ESTADO (modelo + caché LoRA) -------------------------------------- #
# --------------------------------------------------------------------------- #
class ModelState:
    """Mantiene el modelo base (CPU) y caché de LoRA cargados en GPU."""
    def __init__(self) -> None:
        self.base = base_model
        self.active_lora: Optional[str] = None
        self.lora_cache: Dict[str, PeftModel] = {}

    def _to_cpu(self, mdl):
        mdl.to("cpu")
        torch.cuda.empty_cache()

    def _to_gpu(self, mdl):
        mdl.to("cuda")
        torch.cuda.empty_cache()

    def get_active_model(self) -> Any:
        """Devuelve el modelo (en GPU) listo para inferencia."""
        if self.active_lora:
            model = self.lora_cache[self.active_lora]
            self._to_gpu(model)
            return model
        self._to_gpu(self.base)
        return self.base

    def switch_lora(self, lora_path: Optional[str]) -> str:
        """Cambia el LoRA activo (o vuelve al modelo base)."""
        if not lora_path:
            if self.active_lora:
                self._to_cpu(self.lora_cache[self.active_lora])
                self.active_lora = None
            return "✅ Modelo base activo."

        lora_path = str(Path(lora_path).expanduser())
        if not Path(lora_path).exists():
            return f"⚠️ Ruta del LoRA no encontrada: {lora_path}"

        if self.active_lora == lora_path:
            return f"✅ LoRA ya activo: {Path(lora_path).name}"

        # descargamos el LoRA anterior a CPU
        if self.active_lora:
            self._to_cpu(self.lora_cache[self.active_lora])

        # cargar/recuperar LoRA
        if lora_path not in self.lora_cache:
            try:
                logger.info(f"Cargando LoRA desde {lora_path}")
                lora_model = PeftModel.from_pretrained(self.base, lora_path)

                # Activamos Flash‑Attention en el LoRA (si la opción existe)
                if hasattr(lora_model.config, "use_flash_attention_2"):
                    lora_model.config.use_flash_attention_2 = True
                # y nos aseguramos de desactivarlo en el encoder de texto
                if hasattr(lora_model, "text_encoder"):
                    if hasattr(lora_model.text_encoder.config, "use_flash_attention_2"):
                        lora_model.text_encoder.config.use_flash_attention_2 = False

                self.lora_cache[lora_path] = lora_model
            except Exception as e:
                logger.error(f"Error al cargar LoRA: {e}")
                return f"❌ No se pudo cargar el LoRA: {e}"
        else:
            logger.info(f"Reutilizando LoRA cacheado: {lora_path}")

        self.active_lora = lora_path
        return f"✅ LoRA activo: {Path(lora_path).name}"


model_state = ModelState()

# --------------------------------------------------------------------------- #
# GESTOR DE PROMPTS (JSON) -------------------------------------------------- #
# --------------------------------------------------------------------------- #
class PromptManager:
    def __init__(self, file_path: str = "saved_prompts.json"):
        self.file_path = Path(file_path)
        self.prompts: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if self.file_path.is_file():
            try:
                return json.load(self.file_path.open("r", encoding="utf-8"))
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error leyendo prompts: {e}")
        return []

    def _save(self) -> None:
        try:
            with self.file_path.open("w", encoding="utf-8") as f:
                json.dump(self.prompts, f, indent=4, ensure_ascii=False)
        except IOError as e:
            logger.error(f"Error guardando prompts: {e}")

    def names(self) -> List[str]:
        return sorted(p["name"] for p in self.prompts)

    def get(self, name: str) -> str:
        for p in self.prompts:
            if p["name"] == name:
                return p["text"]
        return ""

    def upsert(self, name: str, text: str) -> str:
        name, text = name.strip(), text.strip()
        if not name or not text:
            return "⚠️ Nombre y texto no pueden estar vacíos."
        for p in self.prompts:
            if p["name"] == name:
                p["text"] = text
                p["updated"] = datetime.now().isoformat()
                break
        else:
            self.prompts.append(
                {"name": name, "text": text, "created": datetime.now().isoformat()}
            )
        self._save()
        return f"✅ Prompt '{name}' guardado."

    def delete(self, name: str) -> str:
        before = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["name"] != name]
        if len(self.prompts) < before:
            self._save()
            return f"✅ Prompt '{name}' eliminado."
        return f"⚠️ Prompt '{name}' no encontrado."


prompt_manager = PromptManager()

# --------------------------------------------------------------------------- #
# INTEGRACIÓN CON OLLAMA ---------------------------------------------------- #
# --------------------------------------------------------------------------- #
class OllamaIntegration:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def list_models(self) -> List[str]:
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if r.status_code == 200:
                return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException as e:
            logger.error(f"Ollama no responde: {e}")
        return ["Error: No se pudo conectar a Ollama"]

    def unload(self, model_name: str) -> str:
        if not model_name or "Error" in model_name:
            return "⚠️ Selecciona un modelo válido antes de descargar."
        try:
            subprocess.run(
                ["ollama", "stop", model_name],
                capture_output=True,
                text=True,
                check=True,
            )
            return f"✅ Modelo '{model_name}' descargado."
        except subprocess.CalledProcessError:
            return f"⚠️ No se pudo descargar '{model_name}'."

    def enhance(
        self,
        model_name: str,
        base_prompt: str,
        use_ctx: bool,
        dataset_dir: str,
    ) -> str:
        if not base_prompt.strip():
            return "⚠️ El prompt base está vacío."

        ctx = ""
        if use_ctx and Path(dataset_dir).is_dir():
            meta_path = Path(dataset_dir) / "metadata.jsonl"
            if meta_path.is_file():
                try:
                    words = set()
                    with meta_path.open("r", encoding="utf-8") as f:
                        for line in f:
                            try:
                                data = json.loads(line)
                                words.update(
                                    re.findall(r"\b\w+\b", data.get("text", "").lower())
                                )
                            except json.JSONDecodeError:
                                continue
                    if words:
                        ctx = (
                            "Inspírate en estas palabras clave del dataset: "
                            + ", ".join(sorted(words))
                            + ". "
                        )
                except Exception as e:
                    logger.error(f"Error leyendo metadata: {e}")

        prompt = (
            f"You are an expert music‑prompt writer. {ctx}"
            f"Improve and translate this idea into a concise English prompt for MusicGen: \"{base_prompt}\""
        )
        payload = {"model": model_name, "prompt": prompt, "stream": False}
        try:
            r = self.session.post(
                f"{self.base_url}/api/generate", json=payload, timeout=60
            )
            r.raise_for_status()
            data = r.json()
            out = data.get("response", "").strip().replace('"', "")
            return out[0].upper() + out[1:] if out else base_prompt
        except requests.RequestException as e:
            logger.error(f"Ollama request error: {e}")
            return "⚠️ Error al conectar con Ollama."


ollama = OllamaIntegration()
available_ollama_models = ollama.list_models()
if not settings["ollama_model"] and available_ollama_models:
    settings["ollama_model"] = available_ollama_models[0]

# --------------------------------------------------------------------------- #
# AUDIO TAGGER – Integración con captioning.py ----------------------------- #
# --------------------------------------------------------------------------- #
# Instanciamos una única vez el tagger y cargamos su modelo.
tagger = AudioTagger()
tagger_loaded = False
try:
    load_status = tagger.load_model()
    logger.info(load_status)
    if "Error" in load_status:
        logger.warning(
            "⚠️ No se pudo cargar el modelo de audio‑tagging. "
            "Se usarán placeholders en metadata.jsonl."
        )
    else:
        tagger_loaded = True
except Exception as e:
    logger.error(f"Error fatal al inicializar AudioTagger: {e}")

# --------------------------------------------------------------------------- #
# UTILIDADES PARA EXTRAER LA CAPTION ---------------------------------------- #
# --------------------------------------------------------------------------- #
def _extract_caption(result: Dict[str, Any]) -> str:
    """
    Dada la salida cruda del AudioTagger devuelve una cadena legible.
    Se buscan varios campos habituales (caption, label, labels, tags,
    predictions) y se unen en caso de listas.
    Si no se encuentra nada se devuelve un placeholder genérico.
    """
    # 1️⃣ Campo explícito `caption`
    if "caption" in result and result["caption"]:
        return str(result["caption"]).strip()

    # 2️⃣ Posibles listas o campos singulares
    for key in ("label", "labels", "tags", "predictions", "prediction"):
        if key in result and result[key]:
            val = result[key]
            if isinstance(val, (list, tuple)):
                return ", ".join(map(str, val))
            return str(val).strip()

    # 3️⃣ Como último recurso intentar obtener algo de `description`
    if "description" in result and result["description"]:
        return str(result["description"]).strip()

    return "audio caption placeholder"


# --------------------------------------------------------------------------- #
# FUNCIÓN DE METADATA ------------------------------------------------------- #
# --------------------------------------------------------------------------- #
def generate_metadata(dataset_dir: str) -> str:
    """
    Recorre `dataset_dir` y escribe `metadata.jsonl` usando el
    sistema de captioning de `captioning.py`.

    Cada línea del archivo tiene la forma:
        {"text": "<caption>", "audio_filepath": "<ruta>"}
    """
    if not tagger_loaded:
        return "❌ El modelo de audio‑tagging no está disponible. Revisa los logs."

    root = Path(dataset_dir)
    if not root.is_dir():
        return f"❌ Ruta de dataset inválida: {dataset_dir}"

    # formatos soportados por librosa (usado internamente por el tagger)
    exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
    audio_files = sorted([p for p in root.iterdir() if p.suffix.lower() in exts])
    if not audio_files:
        return f"⚠️ No se encontraron archivos de audio en {dataset_dir}"

    out_path = root / "metadata.jsonl"
    with out_path.open("w", encoding="utf-8") as out_f:
        for audio_path in tqdm.tqdm(
            audio_files, desc="Generando metadata con captioning.py"
        ):
            # API del nuevo AudioTagger devuelve un diccionario.
            result = tagger.process_audio_file(str(audio_path))

            # Registro detallado para depuración (se mostrará sólo en DEBUG)
            logger.debug(f"Resultado raw del tagger para {audio_path.name}: {result}")

            if "error" in result:
                logger.warning(
                    f"Error procesando {audio_path.name}: {result['error']}"
                )
                caption = "audio caption placeholder"
                audio_fp = str(audio_path)
            else:
                caption = _extract_caption(result)
                # Si el tagger ya incluye la ruta, la usamos; si no, usamos la que ya
                # conocemos.
                audio_fp = result.get("file_path", str(audio_path))

            caption_data = {"text": caption, "audio_filepath": audio_fp}
            out_f.write(json.dumps(caption_data, ensure_ascii=False) + "\n")

    return f"✅ metadata.jsonl creado en: {out_path}"


# --------------------------------------------------------------------------- #
# ENTRENAMIENTO (DreamBooth) – sin cambios ---------------------------------- #
# --------------------------------------------------------------------------- #
def _edit_script(script_path: Path, r: int, alpha: int) -> Tuple[bool, str]:
    """Modifica `r=` y `lora_alpha=` dentro del script DreamBooth."""
    if not script_path.is_file():
        return False, f"❌ No se encontró {script_path}"
    try:
        txt = script_path.read_text(encoding="utf-8")
        txt = re.sub(r"r=\d+", f"r={r}", txt)
        txt = re.sub(r"lora_alpha=\d+", f"lora_alpha={alpha}", txt)
        script_path.write_text(txt, encoding="utf-8")
        return True, f"✅ Script actualizado: r={r}, lora_alpha={alpha}"
    except Exception as e:
        logger.error(f"Error editando script: {e}")
        return False, f"❌ Error editando script: {e}"


def launch_training(
    dataset_path: str,
    output_dir: str,
    epochs: int,
    lr: float,
    r: int,
    lora_alpha: int,
    max_duration: int,
    train_seed: int,
) -> Generator[str, None, None]:
    """Lanza `accelerate launch` con los parámetros indicados."""
    script = Path("./musicgen-dreamboothing/dreambooth_musicgen.py")
    ok, msg = _edit_script(script, r, lora_alpha)
    yield msg
    if not ok:
        return

    cmd = [
        "accelerate",
        "launch",
        "dreambooth_musicgen.py",
        f"--model_name_or_path={MODEL_ID}",
        f"--dataset_name={dataset_path}",
        f"--output_dir={output_dir}",
        f"--num_train_epochs={epochs}",
        "--use_lora",
        f"--learning_rate={lr}",
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--fp16",
        "--text_column_name=text",
        "--target_audio_column_name=audio_filepath",
        "--train_split_name=train",
        "--overwrite_output_dir",
        "--do_train",
        "--decoder_start_token_id=2048",
        f"--max_duration_in_seconds={max_duration}",
        "--gradient_checkpointing",
        f"--seed={train_seed}",
    ]

    cwd = Path("./musicgen-dreamboothing")
    logger.info(f"Lanzando entrenamiento: {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
        )
        for line in iter(proc.stdout.readline, ""):
            yield line
        proc.wait()
        if proc.returncode == 0:
            yield "\n✅ Entrenamiento finalizado con éxito."
        else:
            yield f"\n❌ Proceso terminó con código {proc.returncode}."
    except Exception as e:
        logger.error(f"Error lanzando entrenamiento: {e}")
        yield f"❌ Error al lanzar entrenamiento: {e}"


# --------------------------------------------------------------------------- #
# INFERENCIA (generación de música) – sin cambios --------------------------- #
# --------------------------------------------------------------------------- #
def generate_music(
    _: Dict[str, Any],
    lora_path: str,
    prompt: str,
    duration: int,
    seed: int,
    guidance: float,
    temperature: float,
    top_k: int,
    top_p: float,
) -> Tuple[Dict[str, Any], str, Tuple[int, Any]]:
    """Genera audio a partir de un prompt (con o sin LoRA)."""
    status_msg = model_state.switch_lora(lora_path.strip() or None)

    if seed is not None and int(seed) != -1:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        logger.info(f"Semilla fija: {seed}")

    active = model_state.get_active_model()
    active.eval()

    with torch.no_grad():
        inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
        top_k = int(top_k) if top_k > 0 else 250
        top_p_val = top_p if top_p > 0 else None

        logger.info(
            f"Generando audio → prompt='{prompt[:30]}…', dur={duration}s, "
            f"guidance={guidance}, temp={temperature}"
        )
        audio = active.generate(
            **inputs,
            max_new_tokens=int(duration * 50),
            do_sample=True,
            guidance_scale=guidance,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p_val,
        )

    # Liberamos VRAM del modelo que acabamos de usar
    model_state._to_cpu(active)

    sr = base_model.config.audio_encoder.sampling_rate
    audio_np = audio.cpu().numpy().squeeze()
    logger.info("✅ Audio generado, VRAM liberada")

    new_state = {"lora_path": model_state.active_lora}
    return new_state, status_msg, (sr, audio_np)


# --------------------------------------------------------------------------- #
# INTERFAZ GRADIO – sin cambios funcionales ------------------------------- #
# --------------------------------------------------------------------------- #
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎶 Interfaz de Entrenamiento y Generación – MusicGen v4.3")

    active_state = gr.State(value={"lora_path": None})

    with gr.Tabs() as tabs:
        # ------------------------------------------------------------------- #
        # 1️⃣ Entrenamiento LoRA
        # ------------------------------------------------------------------- #
        with gr.TabItem("🛠️ Entrenamiento LoRA"):
            with gr.Row():
                dataset_path_tb = gr.Textbox(
                    label="📂 Ruta al dataset (carpeta con audios)",
                    value=settings["dataset_path"],
                )
                generate_metadata_btn = gr.Button(
                    "🤖 Generar metadata.jsonl (con captioning.py)"
                )
                metadata_out = gr.Textbox(
                    label="Resultado", lines=2, interactive=False
                )

            with gr.Row():
                output_dir_tb = gr.Textbox(
                    label="📁 Carpeta de salida (LoRA)",
                    value=settings["output_dir"],
                )
                epochs_slider = gr.Slider(
                    label="Épocas",
                    minimum=1,
                    maximum=100,
                    step=1,
                    value=settings["epochs"],
                )
                lr_number = gr.Number(label="Learning Rate", value=settings["lr"])

            with gr.Row():
                max_dur_slider = gr.Slider(
                    label="Duración máx. del audio (s)",
                    minimum=10,
                    maximum=300,
                    step=1,
                    value=settings["max_duration"],
                )
                r_slider = gr.Slider(
                    label="R (rank)",
                    minimum=4,
                    maximum=128,
                    step=4,
                    value=settings["lora_r"],
                )
                alpha_slider = gr.Slider(
                    label="Alpha",
                    minimum=4,
                    maximum=256,
                    step=4,
                    value=settings["lora_alpha"],
                )

            train_seed_nb = gr.Number(
                label="Semilla (entrenamiento)",
                value=settings["train_seed"],
                precision=0,
            )
            launch_train_btn = gr.Button(
                "🚀 Lanzar entrenamiento", variant="primary"
            )
            train_log = gr.Textbox(
                label="Log del entrenamiento", lines=15, interactive=False
            )

        # ------------------------------------------------------------------- #
        # 2️⃣ Gestor de Prompts
        # ------------------------------------------------------------------- #
        with gr.TabItem("✍️ Gestor de Prompts"):
            with gr.Row():
                saved_dd = gr.Dropdown(
                    label="Prompts guardados", choices=prompt_manager.names()
                )
                prompt_name_tb = gr.Textbox(label="Nombre del prompt (para guardar)")
                save_btn = gr.Button("💾 Guardar/Actualizar")
                del_btn = gr.Button("🗑️ Eliminar")
                prompt_status_tb = gr.Textbox(label="Estado", interactive=False)

            prompt_tb = gr.Textbox(label="Texto del Prompt", lines=8)

            with gr.Row():
                ollama_dd = gr.Dropdown(
                    label="Modelo Ollama",
                    choices=available_ollama_models,
                    value=settings["ollama_model"],
                )
                unload_ollama_btn = gr.Button("🗑️ Descargar modelo Ollama")
                use_captions_cb = gr.Checkbox(
                    label="Usar captions del dataset como contexto",
                    info="Lee `metadata.jsonl` de la pestaña entrenamiento.",
                )
                enhance_btn = gr.Button(
                    "🔧 Mejorar con Ollama", variant="primary"
                )

        # ------------------------------------------------------------------- #
        # 3️⃣ Generador (Inferencia)
        # ------------------------------------------------------------------- #
        with gr.TabItem("🎵 Generador (Inferencia)"):
            with gr.Row():
                prompt_in_tb = gr.Textbox(
                    label="Prompt musical",
                    placeholder="Ej: Un solo de piano clásico...",
                    value=settings["inference_prompt"],
                    lines=2,
                )
                lora_path_tb = gr.Textbox(
                    label="📂 Ruta al LoRA (vacío = modelo base)",
                    placeholder="./musicgen-dreamboothing/mi_lora_final",
                    value=settings["lora_path"],
                )
                seed_in_nb = gr.Number(
                    label="Semilla (-1 = aleatoria)",
                    value=settings["inference_seed"],
                    precision=0,
                )
                generate_btn = gr.Button("🎹 Generar", variant="primary")
                inference_status = gr.Textbox(
                    label="Estado del modelo",
                    interactive=False,
                    value="✅ Modelo Base Activo",
                )

            duration_slider = gr.Slider(
                label="Duración (s)",
                minimum=5,
                maximum=60,
                step=1,
                value=settings["inference_duration"],
            )
            with gr.Accordion("Ajustes avanzados", open=False):
                guidance_slider = gr.Slider(
                    label="Guidance Scale (CFG)",
                    minimum=1.0,
                    maximum=20.0,
                    step=0.5,
                    value=settings["guidance_scale"],
                )
                temperature_slider = gr.Slider(
                    label="Temperatura",
                    minimum=0.1,
                    maximum=2.0,
                    step=0.05,
                    value=settings["temperature"],
                )
                topk_slider = gr.Slider(
                    label="Top‑k (0 = default 250)",
                    minimum=0,
                    maximum=500,
                    step=10,
                    value=settings["top_k"],
                )
                topp_slider = gr.Slider(
                    label="Top‑p (0 = desactivado)",
                    minimum=0.0,
                    maximum=1.0,
                    step=0.05,
                    value=settings["top_p"],
                )
            audio_out = gr.Audio(label="Resultado", type="numpy")

    # ------------------------------------------------------------------- #
    # CALLBACKS ----------------------------------------------------------- #
    # ------------------------------------------------------------------- #
    def load_prompt(name: str) -> str:
        return prompt_manager.get(name)

    saved_dd.change(fn=load_prompt, inputs=saved_dd, outputs=prompt_tb)

    def save_prompt(name: str, txt: str) -> Tuple[str, List[str]]:
        msg = prompt_manager.upsert(name, txt)
        return msg, prompt_manager.names()

    save_btn.click(
        fn=save_prompt,
        inputs=[prompt_name_tb, prompt_tb],
        outputs=[prompt_status_tb, saved_dd],
    )

    def delete_prompt(name: str) -> Tuple[str, List[str]]:
        msg = prompt_manager.delete(name)
        return msg, prompt_manager.names()

    del_btn.click(fn=delete_prompt, inputs=prompt_name_tb, outputs=[prompt_status_tb, saved_dd])

    unload_ollama_btn.click(fn=ollama.unload, inputs=ollama_dd, outputs=prompt_status_tb)

    def enhance_prompt(model_name: str, base: str, use_ctx: bool, dataset_dir: str) -> str:
        return ollama.enhance(model_name, base, use_ctx, dataset_dir)

    enhance_btn.click(
        fn=enhance_prompt,
        inputs=[ollama_dd, prompt_tb, use_captions_cb, dataset_path_tb],
        outputs=prompt_tb,
    )

    generate_metadata_btn.click(
        fn=generate_metadata, inputs=dataset_path_tb, outputs=metadata_out
    )

    launch_train_btn.click(
        fn=launch_training,
        inputs=[
            dataset_path_tb,
            output_dir_tb,
            epochs_slider,
            lr_number,
            r_slider,
            alpha_slider,
            max_dur_slider,
            train_seed_nb,
        ],
        outputs=train_log,
    )

    def _update_state(lora_path: str) -> Tuple[Dict[str, Any], str]:
        msg = model_state.switch_lora(lora_path.strip() or None)
        return {"lora_path": model_state.active_lora}, msg

    lora_path_tb.submit(fn=_update_state, inputs=lora_path_tb, outputs=[active_state, inference_status])

    generate_btn.click(
        fn=generate_music,
        inputs=[
            active_state,
            lora_path_tb,
            prompt_in_tb,
            duration_slider,
            seed_in_nb,
            guidance_slider,
            temperature_slider,
            topk_slider,
            topp_slider,
        ],
        outputs=[active_state, inference_status, audio_out],
    )

    # ------------------------------------------------------------------- #
    # Persistencia en `settings.json` en tiempo real
    # ------------------------------------------------------------------- #
    def _persist(key: str, val: Any) -> None:
        settings[key] = val
        save_settings(settings)

    component_key_map = [
        (dataset_path_tb, "dataset_path"),
        (output_dir_tb, "output_dir"),
        (epochs_slider, "epochs"),
        (lr_number, "lr"),
        (max_dur_slider, "max_duration"),
        (r_slider, "lora_r"),
        (alpha_slider, "lora_alpha"),
        (train_seed_nb, "train_seed"),
        (prompt_in_tb, "inference_prompt"),
        (duration_slider, "inference_duration"),
        (lora_path_tb, "lora_path"),
        (seed_in_nb, "inference_seed"),
        (guidance_slider, "guidance_scale"),
        (temperature_slider, "temperature"),
        (topk_slider, "top_k"),
        (topp_slider, "top_p"),
    ]
    for comp, key in component_key_map:
        comp.change(fn=lambda v, k=key: _persist(k, v), inputs=comp, outputs=None)

if __name__ == "__main__":
    demo.queue()
    demo.launch(share=False, debug=False)
