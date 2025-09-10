# app.py (Versión 4.0 Final - Arquitectura de Estado Corregida)
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

from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

# --- Configuración de Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Clases de Lógica (sin cambios) ---
class PromptManager:
    """Gestor para guardar, cargar y eliminar prompts."""
    def __init__(self, prompts_file="saved_prompts.json"):
        self.prompts_file = prompts_file
        self.prompts = self.load_prompts()
    def load_prompts(self):
        if os.path.exists(self.prompts_file):
            try:
                with open(self.prompts_file, 'r', encoding='utf-8') as f: return json.load(f)
            except (json.JSONDecodeError, IOError): return []
        return []
    def save_prompts(self):
        try:
            with open(self.prompts_file, 'w', encoding='utf-8') as f: json.dump(self.prompts, f, indent=4, ensure_ascii=False)
        except IOError as e: logger.error(f"Error guardando prompts: {e}")
    def update_prompt(self, name, text):
        found = False
        for prompt in self.prompts:
            if prompt["name"] == name:
                prompt["text"] = text; found = True; break
        if not found and name.strip() and text.strip():
            self.prompts.append({"name": name, "text": text, "created": datetime.now().isoformat()})
        self.save_prompts()
        return f"Prompt '{name}' guardado/actualizado."
    def delete_prompt(self, name):
        initial_len = len(self.prompts)
        self.prompts = [p for p in self.prompts if p["name"] != name]
        if len(self.prompts) < initial_len:
            self.save_prompts(); return f"Prompt '{name}' eliminado."
        return f"Prompt '{name}' no encontrado."
    def get_prompt_text(self, name):
        for prompt in self.prompts:
            if prompt["name"] == name: return prompt["text"]
        return ""
    def get_prompt_names(self):
        return sorted([p["name"] for p in self.prompts])

class OllamaIntegration:
    """Integración con Ollama para mejorar y traducir prompts."""
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url
        self.session = requests.Session()
    def get_available_models(self):
        try:
            response = self.session.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200: return [model["name"] for model in response.json().get("models", [])]
            return []
        except requests.ConnectionError: return ["Error: No se pudo conectar a Ollama"]
        except Exception as e: logger.error(f"Error desconocido de Ollama: {e}"); return []
    def unload_ollama_model(self, model_name: str):
        if not model_name or "Error" in model_name: return "Selecciona un modelo válido para descargar."
        try:
            subprocess.run(["ollama", "stop", model_name], capture_output=True, text=True, check=True)
            return f"✅ Modelo '{model_name}' descargado de la memoria."
        except Exception as e:
            return f"⚠️ No se pudo descargar '{model_name}' (quizás no estaba cargado)."
    def enhance_and_translate_prompt(self, model_name: str, base_prompt: str, use_captions_context: bool, dataset_path: str):
        if not base_prompt.strip(): return "El prompt base no puede estar vacío."
        tags_context = ""
        if use_captions_context and dataset_path and os.path.isdir(dataset_path):
            metadata_file = os.path.join(dataset_path, "metadata.jsonl")
            if os.path.exists(metadata_file):
                try:
                    all_words = set()
                    with open(metadata_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            data = json.loads(line)
                            words = re.findall(r'\b\w+\b', data.get("text", "").lower())
                            all_words.update(words)
                    if all_words: tags_context = f"To better align with the user's music library, draw inspiration from these keywords: {', '.join(sorted(list(all_words)))}. "
                except Exception as e: logger.error(f"Error al leer metadata.jsonl: {e}")
        try:
            ollama_prompt = f'''You are an expert music prompt writer... (omitting for brevity)... Translate and enhance the following user idea: "{base_prompt}"'''
            response = self.session.post(f"{self.base_url}/api/generate", json={"model": model_name, "prompt": ollama_prompt, "stream": False}, timeout=60)
            if response.status_code == 200:
                result = response.json()
                enhanced_prompt = result.get("response", "").strip().replace('"', '')
                return enhanced_prompt[0].upper() + enhanced_prompt[1:] if enhanced_prompt else base_prompt
            else: return f"Error de Ollama: {response.status_code}"
        except Exception as e:
            logger.error(f"Error mejorando prompt: {e}"); return "Error al conectar con Ollama."

# --- LÓGICA DE LA INTERFAZ ---
SETTINGS_FILE = "settings.json"
MODEL_ID = "facebook/musicgen-small"
prompt_manager = PromptManager()
ollama = OllamaIntegration()
available_ollama_models = ollama.get_available_models()

def load_settings():
    defaults = {"dataset_path": "", "output_dir": "./mi_lora_final", "epochs": 15, "lr": 0.0001, "lora_r": 32, "lora_alpha": 64, "max_duration": 180, "ollama_model": available_ollama_models[0] if available_ollama_models else "", "train_seed": 42, "inference_prompt": "80s rock ballad with a power guitar solo", "inference_duration": 15, "lora_path": "", "inference_seed": -1, "guidance_scale": 3.0, "temperature": 1.0, "top_k": 250, "top_p": 0.0}
    if os.path.exists(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE, 'r') as f:
                settings = json.load(f)
                defaults.update(settings)
                return defaults
        except (json.JSONDecodeError, IOError): pass
    return defaults

# --- CARGA INICIAL DE MODELOS (SOLO UNA VEZ) ---
print("Cargando procesador y modelo base...")
settings = load_settings()
processor = AutoProcessor.from_pretrained(MODEL_ID)
# Se carga en la CPU para no ocupar VRAM al inicio.
base_model = MusicgenForConditionalGeneration.from_pretrained(MODEL_ID, torch_dtype=torch.float16)
base_model.to("cpu")
print("¡Listo!")

# --- FUNCIONES DE LA INTERFAZ (ARQUITECTURA CORRECTA) ---

def switch_model_and_state(lora_path):
    """Carga un LoRA o vuelve al modelo base. Devuelve el nuevo objeto de estado y un mensaje."""
    print(f"Cambiando modelo. Ruta LoRA: {lora_path}")
    base_model.to("cuda") # Mover a GPU para aplicar el LoRA
    if lora_path and os.path.exists(lora_path):
        try:
            active_model = PeftModel.from_pretrained(base_model, lora_path)
            status = f"✅ LoRA Activo: {os.path.basename(lora_path)}"
            new_state = {"model": active_model, "lora_path": lora_path}
        except Exception as e:
            print(f"Error cargando LoRA: {e}")
            active_model = base_model
            status = f"❌ Error al cargar LoRA: {e}"
            new_state = {"model": base_model, "lora_path": None}
    else:
        active_model = base_model
        status = "✅ Modelo Base Activo"
        new_state = {"model": base_model, "lora_path": None}
    
    # Devolver el modelo a la CPU para liberar VRAM hasta que se use
    active_model.to("cpu")
    torch.cuda.empty_cache()
    print(status)
    return new_state, status

def generate_music_with_state(current_state, lora_path_textbox, prompt, duration, seed, guidance, temp, topk, topp):
    """Función principal de generación que gestiona el estado del modelo."""
    active_model = current_state["model"]
    current_lora = current_state["lora_path"]
    
    # Comprobar si el LoRA deseado es diferente al que está cargado en el estado
    if lora_path_textbox != current_lora:
        new_state, status = switch_model_and_state(lora_path_textbox)
        active_model = new_state["model"]
    else:
        status = f"✅ LoRA Activo: {os.path.basename(current_lora)}" if current_lora else "✅ Modelo Base Activo"
        new_state = current_state

    # Mover a GPU para generar
    active_model.to("cuda")
    print("Generando audio en GPU...")
    
    if seed is not None and int(seed) != -1:
        torch.manual_seed(int(seed))
        torch.cuda.manual_seed_all(int(seed))
        
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    if topk == 0 and topp == 0.0: topk = 250
    
    audio_values = active_model.generate(**inputs, max_new_tokens=int(duration * 50), do_sample=True, guidance_scale=guidance, temperature=temp, top_k=int(topk), top_p=topp if topp > 0 else None)
    
    # Devolver a CPU para liberar VRAM
    active_model.to("cpu")
    torch.cuda.empty_cache()
    print("Generación completada, VRAM liberada.")
    
    return new_state, status, (base_model.config.audio_encoder.sampling_rate, audio_values.cpu().numpy().squeeze())

# (El resto de funciones de entrenamiento y prompts no necesitan cambios)
def modify_and_run_training(dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, train_seed):
    # (El código de esta función es el mismo de la versión anterior)
    script_path = "./musicgen-dreamboothing/dreambooth_musicgen.py"
    try:
        with open(script_path, 'r', encoding='utf-8') as f: script_content = f.read()
        script_content = re.sub(r'r=\d+', f'r={int(lora_r)}', script_content)
        script_content = re.sub(r'lora_alpha=\d+', f'lora_alpha={int(lora_alpha)}', script_content)
        with open(script_path, 'w', encoding='utf-8') as f: f.write(script_content)
        yield f"✅ Script modificado: r={int(lora_r)}, lora_alpha={int(lora_alpha)}\n"
    except Exception as e:
        yield f"❌ Error al modificar script: {e}"; return

    command = ["accelerate", "launch", "dreambooth_musicgen.py", "--model_name_or_path=facebook/musicgen-small", f"--dataset_name={dataset_path}", f"--output_dir={output_dir}", f"--num_train_epochs={int(epochs)}", "--use_lora", f"--learning_rate={lr}", "--per_device_train_batch_size=1", "--gradient_accumulation_steps=4", "--fp16", "--text_column_name=description", "--target_audio_column_name=audio_filepath", "--train_split_name=train", "--overwrite_output_dir", "--do_train", "--decoder_start_token_id=2048", f"--max_duration_in_seconds={int(max_duration)}", "--gradient_checkpointing", f"--seed={int(train_seed)}"]
    yield f"🚀 Lanzando entrenamiento...\n\n"
    process = subprocess.Popen(command, cwd="./musicgen-dreamboothing", stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    process.wait()
    yield output + "\n\n✅ ¡Entrenamiento finalizado!"
def on_select_prompt(name): return prompt_manager.get_prompt_text(name)
def on_save_prompt(name, text):
    msg = prompt_manager.update_prompt(name, text)
    return msg, gr.Dropdown(choices=prompt_manager.get_prompt_names(), value=name)
def on_delete_prompt(name):
    msg = prompt_manager.delete_prompt(name)
    return msg, gr.Dropdown(choices=prompt_manager.get_prompt_names(), value=None)

# --- DISEÑO DE LA INTERFAZ ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎶 Interfaz de Entrenamiento y Generación para MusicGen v4.0")
    
    # --- Componente de Estado para el Modelo de Inferencia ---
    # Guarda un diccionario con el modelo activo y la ruta del LoRA cargado.
    initial_state = {"model": base_model, "lora_path": None}
    active_model_state = gr.State(value=initial_state)
    
    with gr.Tabs(elem_id="tabs") as tabs:
        with gr.TabItem("🛠️ Entrenar LoRA", id=0):
            # ... (UI de entrenamiento sin cambios)
            with gr.Group():
                prep_dataset_path_input = gr.Textbox(label="Ruta a la Carpeta con tus Audios", value=settings.get("dataset_path"))
                gr.Markdown("Para generar/mejorar `metadata.jsonl`, ve a la pestaña 'Gestor de Prompts'.")
            with gr.Group():
                gr.Markdown("### Configurar y Lanzar Entrenamiento")
                output_dir_input = gr.Textbox(label="Nombre de la Carpeta para el LoRA", value=settings.get("output_dir"))
                with gr.Row():
                    epochs_input = gr.Slider(label="Épocas", minimum=1, maximum=100, value=settings.get("epochs"), step=1)
                    lr_input = gr.Number(label="Learning Rate", value=settings.get("lr"))
                    train_seed_input = gr.Number(label="Seed de Entrenamiento", value=settings.get("train_seed"), precision=0)
                with gr.Row():
                    max_duration_input = gr.Slider(label="Duración Máx. Audio (s)", minimum=10, maximum=300, value=settings.get("max_duration"), step=1)
                    lora_r_input = gr.Slider(label="Rango (r)", minimum=4, maximum=128, value=settings.get("lora_r"), step=4)
                    lora_alpha_input = gr.Slider(label="Alpha", minimum=4, maximum=256, value=settings.get("lora_alpha"), step=4)
                launch_train_button = gr.Button("🚀 ¡Lanzar Entrenamiento!", variant="primary")
                train_output = gr.Textbox(label="Registro del Entrenamiento", interactive=False, lines=15)
        
        with gr.TabItem("✍️ Gestor de Prompts", id=1):
            # ... (UI de gestor de prompts sin cambios)
            gr.Markdown("## Gestor y Mejorador de Prompts con Ollama")
            with gr.Row():
                with gr.Column(scale=1):
                    prompt_select_dd = gr.Dropdown(label="Prompts Guardados", choices=prompt_manager.get_prompt_names())
                    prompt_name_tb = gr.Textbox(label="Nombre del Prompt (para guardar)")
                    prompt_save_btn = gr.Button("💾 Guardar/Actualizar")
                    prompt_delete_btn = gr.Button("🗑️ Eliminar")
                    prompt_status_tb = gr.Textbox(label="Estado del Gestor", interactive=False)
                with gr.Column(scale=2):
                    prompt_text_area = gr.Textbox(label="Texto del Prompt", lines=10, placeholder="Escribe aquí tu idea para un prompt...")
                    use_in_inference_btn = gr.Button("🎵 Usar este Prompt en el Generador")
            with gr.Group():
                gr.Markdown("### ✨ Mejorar Prompt con Ollama")
                with gr.Row():
                    ollama_model_select = gr.Dropdown(label="Modelo de Ollama", choices=available_ollama_models, value=settings.get("ollama_model"), scale=3)
                    unload_ollama_button = gr.Button("🗑️ Descargar Modelo Ollama", scale=1)
                use_captions_checkbox = gr.Checkbox(label="Usar captions del dataset como contexto", info="Lee el metadata.jsonl de la ruta de la pestaña de Entrenamiento.")
                enhance_btn = gr.Button("Mejorar y Traducir Prompt Actual", variant="primary")
                gr.Markdown("*Nota: La primera vez que uses un modelo de Ollama o después de descargarlo, la respuesta tardará más.*")
            with gr.Group():
                 gr.Markdown("### 🗃️ Generador de `metadata.jsonl`")
                 generate_metadata_button = gr.Button("🤖 Generar `metadata.jsonl` (básico)")
                 metadata_output = gr.Textbox(label="Registro de Generación", interactive=False, lines=5)
        
        with gr.TabItem("🎵 Generador de Música (Inferencia)", id=2):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Prompt Musical", placeholder="Ej: Un solo de piano clásico...", value=settings.get("inference_prompt"))
                    duration_input = gr.Slider(minimum=5, maximum=60, value=settings.get("inference_duration"), label="Duración (s)")
                    with gr.Row():
                        lora_path_input = gr.Textbox(label="Ruta al LoRA", placeholder="./musicgen-dreamboothing/mi_lora_final", scale=3, value=settings.get("lora_path"))
                        inference_seed_input = gr.Number(label="Seed (-1 aleatorio)", value=settings.get("inference_seed"), precision=0, scale=1)
                with gr.Column(scale=1):
                    generate_button = gr.Button("🎹 Generar Música", variant="primary")
                    status_output = gr.Textbox(label="Modelo Activo", interactive=False, value="✅ Modelo Base Activo")
            with gr.Accordion("Ajustes Avanzados de Inferencia", open=False):
                with gr.Row():
                    guidance_input = gr.Slider(label="Guidance Scale (CFG)", minimum=1.0, maximum=20.0, value=settings.get("guidance_scale"), step=0.5)
                    temp_input = gr.Slider(label="Temperatura", minimum=0.1, maximum=2.0, value=settings.get("temperature"), step=0.05)
                    topk_input = gr.Slider(label="Top-k", minimum=0, maximum=500, value=settings.get("top_k"), step=10, info="0 para desactivar")
                    topp_input = gr.Slider(label="Top-p (Nucleus)", minimum=0.0, maximum=1.0, value=settings.get("top_p"), step=0.05, info="0 para desactivar")
            audio_output = gr.Audio(label="Resultado Generado")

    # --- Conexiones de la Interfaz ---
    # Gestor de Prompts
    prompt_select_dd.change(fn=on_select_prompt, inputs=prompt_select_dd, outputs=prompt_text_area)
    prompt_save_btn.click(fn=on_save_prompt, inputs=[prompt_name_tb, prompt_text_area], outputs=[prompt_status_tb, prompt_select_dd])
    prompt_delete_btn.click(fn=on_delete_prompt, inputs=[prompt_name_tb], outputs=[prompt_status_tb, prompt_select_dd])
    enhance_btn.click(fn=ollama.enhance_and_translate_prompt, inputs=[ollama_model_select, prompt_text_area, use_captions_checkbox, prep_dataset_path_input], outputs=prompt_text_area)
    unload_ollama_button.click(fn=ollama.unload_ollama_model, inputs=[ollama_model_select], outputs=[prompt_status_tb])
    use_in_inference_btn.click(fn=lambda x: (x, gr.Tabs(selected=2)), inputs=prompt_text_area, outputs=[prompt_input, tabs])
    
    def run_prepare_dataset_wrapper(path):
        tagger = AudioTagger()
        if not tagger.load_model(): return "Error: No se pudo cargar el modelo de tagging."
        if not path or not os.path.isdir(path): return "Error: Ruta de dataset inválida."
        supported_extensions = ['.wav', '.mp3', '.flac', '.ogg']
        audio_files = sorted([f for f in os.listdir(path) if os.path.splitext(f)[1].lower() in supported_extensions])
        if not audio_files: return f"No se encontraron archivos de audio en '{path}'"
        output_file = os.path.join(path, "metadata.jsonl")
        with open(output_file, 'w', encoding='utf-8') as f:
            for audio_file in tqdm(audio_files, desc="Generando metadata.jsonl"):
                full_path = os.path.join(path, audio_file)
                result = tagger.generate_caption_from_file(full_path)
                if "error" not in result: f.write(json.dumps(result, ensure_ascii=False) + '\n')
        return f"✅ ¡Proceso completado! Archivo guardado en:\n{output_file}"
    generate_metadata_button.click(fn=run_prepare_dataset_wrapper, inputs=[prep_dataset_path_input], outputs=[metadata_output])

    # Entrenamiento
    launch_train_button.click(fn=modify_and_run_training, inputs=[prep_dataset_path_input, output_dir_input, epochs_input, lr_input, lora_r_input, lora_alpha_input, max_duration_input, train_seed_input], outputs=[train_output])
    
    # Inferencia (CON GESTIÓN DE ESTADO CORRECTA)
    lora_path_input.submit(
        fn=switch_model_and_state,
        inputs=[lora_path_input],
        outputs=[active_model_state, status_output]
    )
    generate_button.click(
        fn=generate_music_with_state,
        inputs=[active_model_state, lora_path_input, prompt_input, duration_input, inference_seed_input, guidance_input, temp_input, topk_input, topp_input],
        outputs=[active_model_state, status_output, audio_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)