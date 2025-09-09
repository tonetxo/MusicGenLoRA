# app.py (Versión Definitiva con imports corregidos)
import gradio as gr
import torch
import os
import subprocess
import re
import json
import gc
import logging
from datetime import datetime
import requests
from tqdm import tqdm

# --- IMPORTACIONES CORREGIDAS ---
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel
# -----------------------------

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Clases de Lógica (Integradas de tus scripts) ---

class AudioTagger:
    """Componente para etiquetar archivos de audio (lógica de prepare_dataset.py)."""
    def __init__(self, model_name="MIT/ast-finetuned-audioset-10-10-0.4593"):
        self.processor = None
        self.model = None
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self):
        try:
            from transformers import AutoProcessor, AutoModelForAudioClassification
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForAudioClassification.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            return True
        except Exception as e:
            logger.error(f"Error cargando modelo de tagging: {e}")
            return False

    def generate_caption_from_file(self, audio_path: str, top_k: int = 5):
        if not self.model or not self.processor: return {"error": "El modelo de tagging no está cargado."}
        try:
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = self.model(**inputs)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            top_probs, top_indices = torch.topk(probabilities[0], top_k)
            labels = [self.model.config.id2label[idx.item()] for prob, idx in zip(top_probs, top_indices)]
            caption = ", ".join(labels)
            return {"file_name": os.path.basename(audio_path), "text": caption}
        except Exception as e:
            return {"error": f"Error procesando '{audio_path}': {str(e)}"}

class OllamaIntegration:
    """Integración con Ollama para mejorar prompts (lógica de prompting.py)."""
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()

    def get_available_models(self):
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                return [model["name"] for model in models_data.get("models", [])]
            return []
        except Exception:
            return ["Error: No se pudo conectar a Ollama"]

    def unload_ollama_model(self, model_name: str):
        try:
            subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True, timeout=10)
        except Exception as e:
            logger.warning(f"No se pudo descargar el modelo {model_name} de Ollama: {e}")

    def enhance_prompt(self, model_name: str, base_prompt: str, unload_after: bool = True):
        try:
            ollama_prompt = f'''You are an expert audio prompt writer. Your task is to rewrite a basic prompt into a rich, descriptive one.
CRITICAL RULE: Your output MUST be a description of a sound, NOT a command to create it.
Do NOT start with verbs like "Create", "Generate", or "Make". Do NOT use quotation marks.
The output must be a single block of text containing only the description.

- Bad example: "Create a song with a fast beat."
- Good example: A high-energy track with a driving, fast-paced electronic beat, pulsating synth bass, and atmospheric pads.

Rewrite the following prompt: "{base_prompt}"'''
            
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json={"model": model_name, "prompt": ollama_prompt, "stream": False},
                timeout=60
            )
            if response.status_code == 200:
                result = response.json()
                enhanced_prompt = result.get("response", "").strip()
                
                # Post-procesado para limpiar la salida de Ollama
                if enhanced_prompt:
                    enhanced_prompt = re.sub(r'^[\"\'“]?(Create|Generate|Make|Produce)\s+(a|an)\s+', '', enhanced_prompt, flags=re.IGNORECASE)
                    enhanced_prompt = enhanced_prompt.strip('\"\'”')
                    enhanced_prompt = enhanced_prompt[0].upper() + enhanced_prompt[1:]
                return enhanced_prompt
            else:
                logger.error(f"Error en la respuesta de Ollama: {response.status_code}")
                return base_prompt
        except Exception as e:
            logger.error(f"Error mejorando prompt con Ollama: {e}")
            return base_prompt

# --- Lógica de la Interfaz ---
SETTINGS_FILE = "settings.json"
MODEL_ID = "facebook/musicgen-small"
ollama = OllamaIntegration()
available_ollama_models = ollama.get_available_models()

def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as f: return json.load(f)
    return {"dataset_path": "", "output_dir": "./mi_lora_final", "epochs": 15, "lr": 0.0001, "lora_r": 32, "lora_alpha": 64, "max_duration": 180, "ollama_model": available_ollama_models[0] if available_ollama_models else ""}

def save_settings(dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, ollama_model):
    settings = {"dataset_path": dataset_path, "output_dir": output_dir, "epochs": epochs, "lr": lr, "lora_r": lora_r, "lora_alpha": lora_alpha, "max_duration": max_duration, "ollama_model": ollama_model}
    with open(SETTINGS_FILE, 'w') as f: json.dump(settings, f, indent=4)

print("Cargando ajustes y modelo base...")
settings = load_settings()
processor = AutoProcessor.from_pretrained(MODEL_ID)
base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
model = base_model
current_lora = None
print("¡Listo!")

def free_vram():
    global model, base_model; model = None; base_model = None; gc.collect(); torch.cuda.empty_cache()
    return "✅ VRAM liberada."

def ensure_base_model_is_loaded():
    global base_model, model, current_lora
    if base_model is None:
        base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16).to("cuda")
        model = base_model; current_lora = None
        return True
    return False

def load_lora_model(lora_path):
    global model, current_lora, base_model; ensure_base_model_is_loaded()
    if lora_path and os.path.exists(lora_path):
        try:
            model = PeftModel.from_pretrained(base_model, lora_path); model = model.merge_and_unload(); current_lora = lora_path
            return f"✅ LoRA '{lora_path}' cargado."
        except Exception as e: return f"❌ Error: {e}"
    else:
        model = base_model; current_lora = None; return "ℹ️ Usando modelo base."

def generate_music(prompt, duration_secs, lora_path):
    if ensure_base_model_is_loaded(): load_lora_model(lora_path)
    if lora_path != current_lora: load_lora_model(lora_path)
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    audio_values = model.generate(**inputs, max_new_tokens=int(duration_secs * 50))
    return (model.config.audio_encoder.sampling_rate, audio_values.cpu().numpy().squeeze())

def generate_captions_and_enhance(dataset_path, ollama_model, progress=gr.Progress(track_tqdm=True)):
    if not dataset_path or not os.path.isdir(dataset_path):
        yield "Error: Ruta de dataset inválida."
        return

    yield "Iniciando Fase 1: Generando etiquetas base..."
    tagger = AudioTagger()
    if not tagger.load_model():
        yield "Error: No se pudo cargar el modelo de tagging de audio."
        return

    supported_extensions = ['.wav', '.mp3', '.flac', '.ogg']
    audio_files = sorted([f for f in os.listdir(dataset_path) if os.path.splitext(f)[1].lower() in supported_extensions])
    
    base_captions = []
    for audio_file in tqdm(audio_files, desc="Fase 1: Generando etiquetas"):
        full_path = os.path.join(dataset_path, audio_file)
        result = tagger.generate_caption_from_file(full_path)
        if "error" not in result:
            base_captions.append(result)
    
    yield f"Fase 1 completada. Se generaron {len(base_captions)} etiquetas base. Iniciando Fase 2: Mejora con Ollama..."

    output_file = os.path.join(dataset_path, "metadata.jsonl")
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in tqdm(base_captions, desc="Fase 2: Mejorando con Ollama"):
            enhanced_text = ollama.enhance_prompt(ollama_model, item["text"], unload_after=False)
            item["text"] = enhanced_text
            f.write(json.dumps(item) + '\n')
    
    ollama.unload_ollama_model(ollama_model)
    yield f"✅ ¡Proceso completado! Se ha creado y mejorado el archivo:\n{output_file}"

def modify_and_run_training(dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, ollama_model):
    save_settings(dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, ollama_model)
    script_path = "./musicgen-dreamboothing/dreambooth_musicgen.py"
    try:
        with open(script_path, 'r', encoding='utf-8') as f: script_content = f.read()
        script_content = re.sub(r'r=\d+', f'r={int(lora_r)}', script_content)
        script_content = re.sub(r'lora_alpha=\d+', f'lora_alpha={int(lora_alpha)}', script_content)
        with open(script_path, 'w', encoding='utf-8') as f: f.write(script_content)
        yield f"✅ Script modificado: r={int(lora_r)}, lora_alpha={int(lora_alpha)}\n"
    except Exception as e:
        yield f"❌ Error al modificar script: {e}"; return

    command = ["accelerate", "launch", "dreambooth_musicgen.py", "--model_name_or_path=facebook/musicgen-small", f"--dataset_name={dataset_path}", f"--output_dir={output_dir}", f"--num_train_epochs={int(epochs)}", "--use_lora", f"--learning_rate={lr}", "--per_device_train_batch_size=1", "--gradient_accumulation_steps=4", "--fp16", "--text_column_name=description", "--target_audio_column_name=audio_filepath", "--train_split_name=train", "--overwrite_output_dir", "--do_train", "--decoder_start_token_id=2048", f"--max_duration_in_seconds={int(max_duration)}", "--gradient_checkpointing"]
    yield f"🚀 Lanzando entrenamiento...\n\n"
    process = subprocess.Popen(command, cwd="./musicgen-dreamboothing", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    process.wait()
    yield output + "\n\n✅ ¡Entrenamiento finalizado!"

# --- DISEÑO DE LA INTERFAZ ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎶 Interfaz de Entrenamiento y Generación para MusicGen")
    with gr.Tabs():
        with gr.TabItem("🛠️ Entrenar LoRA"):
            gr.Markdown("## Flujo de Entrenamiento Completo")
            with gr.Group():
                gr.Markdown("### Paso 1: Preparar Dataset")
                prep_dataset_path_input = gr.Textbox(label="Ruta a la Carpeta con tus Audios", value=settings.get("dataset_path"))
                with gr.Accordion("Opciones Avanzadas (Ollama)", open=False):
                    ollama_model_select = gr.Dropdown(label="Modelo de Ollama para mejorar prompts", choices=available_ollama_models, value=settings.get("ollama_model"))
                    enhance_button = gr.Button("🤖✨ Generar y Mejorar `metadata.jsonl` con Ollama")
                prep_output = gr.Textbox(label="Registro del Proceso de Preparación", interactive=False, lines=7)
            with gr.Group():
                gr.Markdown("### Paso 2: Configurar y Lanzar Entrenamiento")
                output_dir_input = gr.Textbox(label="Nombre de la Carpeta para el LoRA", value=settings.get("output_dir"))
                with gr.Row():
                    epochs_input = gr.Slider(label="Épocas", minimum=1, maximum=100, value=settings.get("epochs"), step=1)
                    lr_input = gr.Number(label="Learning Rate", value=settings.get("lr"))
                    max_duration_input = gr.Slider(label="Duración Máx. Audio (s)", minimum=10, maximum=300, value=settings.get("max_duration"), step=1)
                with gr.Row():
                    lora_r_input = gr.Slider(label="Rango (r)", minimum=4, maximum=128, value=settings.get("lora_r"), step=4)
                    lora_alpha_input = gr.Slider(label="Alpha", minimum=4, maximum=256, value=settings.get("lora_alpha"), step=4)
                launch_train_button = gr.Button("🚀 ¡Lanzar Entrenamiento!", variant="primary")
                train_output = gr.Textbox(label="Registro del Entrenamiento", interactive=False, lines=15)
        with gr.TabItem("🎵 Generador de Música (Inferencia)"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Prompt Musical", placeholder="Ej: Un solo de piano clásico...")
                    duration_input = gr.Slider(minimum=5, maximum=60, value=15, label="Duración (s)")
                    lora_path_input = gr.Textbox(label="Ruta al LoRA", placeholder="./musicgen-dreamboothing/mi_lora_final")
                with gr.Column(scale=1):
                    generate_button = gr.Button("🎹 Generar Música", variant="primary")
                    free_vram_button = gr.Button("🗑️ Liberar VRAM")
                    status_output = gr.Textbox(label="Estado", interactive=False, value="ℹ️ Usando modelo base.")
            audio_output = gr.Audio(label="Resultado Generado")
    # --- Conexiones de la Interfaz ---
    enhance_button.click(fn=generate_captions_and_enhance, inputs=[prep_dataset_path_input, ollama_model_select], outputs=[prep_output])
    launch_train_button.click(fn=modify_and_run_training, inputs=[prep_dataset_path_input, output_dir_input, epochs_input, lr_input, lora_r_input, lora_alpha_input, max_duration_input, ollama_model_select], outputs=[train_output])
    generate_button.click(fn=generate_music, inputs=[prompt_input, duration_input, lora_path_input], outputs=[audio_output])
    lora_path_input.submit(fn=load_lora_model, inputs=[lora_path_input], outputs=[status_output])
    free_vram_button.click(fn=free_vram, inputs=[], outputs=[status_output])

if __name__ == "__main__":
    demo.launch(share=False)