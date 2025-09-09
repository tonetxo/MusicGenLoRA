# app.py (Versión Definitiva)
import gradio as gr
import torch
import os
import subprocess
import re
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from peft import PeftModel

# --- CONFIGURACIÓN INICIAL ---
MODEL_ID = "facebook/musicgen-small"
print("Cargando modelo base y procesador para inferencia...")
processor = AutoProcessor.from_pretrained(MODEL_ID)
# No cargamos el modelo en 4-bit aquí para evitar posibles conflictos con bitsandbytes al entrenar
base_model = MusicgenForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16
).to("cuda")
model = base_model
print("Modelo cargado. ¡Listo para generar!")

# --- LÓGICA DE LA APLICACIÓN ---
current_lora = None

def load_lora_model(lora_path):
    global model, current_lora, base_model
    if lora_path and os.path.exists(lora_path):
        try:
            print(f"Cargando adaptador LoRA desde: {lora_path}")
            model = PeftModel.from_pretrained(base_model, lora_path)
            model = model.merge_and_unload()
            current_lora = lora_path
            return f"✅ Adaptador LoRA '{lora_path}' cargado."
        except Exception as e:
            return f"❌ Error al cargar LoRA: {e}"
    else:
        model = base_model
        current_lora = None
        return "ℹ️ Usando modelo base."

def generate_music(prompt, duration_secs, lora_path):
    global model, current_lora
    if lora_path != current_lora:
        status = load_lora_model(lora_path)
        print(status)
    
    print(f"Generando música para: '{prompt}'")
    inputs = processor(text=[prompt], padding=True, return_tensors="pt").to("cuda")
    max_new_tokens = int(duration_secs * 50) 
    audio_values = model.generate(**inputs, max_new_tokens=max_new_tokens)
    sampling_rate = model.config.audio_encoder.sampling_rate
    audio_numpy = audio_values.cpu().numpy().squeeze()
    return (sampling_rate, audio_numpy)

def run_prepare_dataset(dataset_path):
    if not dataset_path or not os.path.isdir(dataset_path):
        yield "Por favor, introduce una ruta válida a tu carpeta de dataset."
        return
    
    command = ["python3", "prepare_dataset.py", dataset_path]
    yield f"🚀 Ejecutando preparación en '{dataset_path}'..."
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding='utf-8')
    
    output = ""
    for line in iter(process.stdout.readline, ''):
        output += line
        yield output
    process.wait()
    yield output + "\n\n✅ ¡Proceso finalizado!"

def modify_and_run_training(
    dataset_path, output_dir, epochs, lr, lora_r, lora_alpha, max_duration, progress=gr.Progress(track_tqdm=True)
):
    # --- Paso 1: Modificar el script de entrenamiento con el rank y alpha deseados ---
    script_path = "./musicgen-dreamboothing/dreambooth_musicgen.py"
    try:
        with open(script_path, 'r', encoding='utf-8') as f:
            script_content = f.read()

        # Usamos expresiones regulares para reemplazar los valores de r y lora_alpha
        script_content = re.sub(r'r=\d+', f'r={int(lora_r)}', script_content)
        script_content = re.sub(r'lora_alpha=\d+', f'lora_alpha={int(lora_alpha)}', script_content)
        
        with open(script_path, 'w', encoding='utf-8') as f:
            f.write(script_content)
        
        yield f"✅ Script modificado: r={int(lora_r)}, lora_alpha={int(lora_alpha)}\n"

    except Exception as e:
        yield f"❌ Error al modificar el script: {e}"
        return

    # --- Paso 2: Construir y ejecutar el comando ---
    command = [
        "accelerate", "launch", "dreambooth_musicgen.py",
        "--model_name_or_path=facebook/musicgen-small",
        f"--dataset_name={dataset_path}",
        f"--output_dir={output_dir}",
        f"--num_train_epochs={int(epochs)}",
        "--use_lora",
        f"--learning_rate={lr}",
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=4",
        "--fp16",
        "--text_column_name=description",
        "--target_audio_column_name=audio_filepath",
        "--train_split_name=train",
        "--overwrite_output_dir",
        "--do_train",
        "--decoder_start_token_id=2048",
        f"--max_duration_in_seconds={int(max_duration)}",
        "--gradient_checkpointing",
    ]
    
    yield f"🚀 Lanzando entrenamiento con el comando:\n{' '.join(command)}\n\n"
    
    # Ejecutamos desde la carpeta correcta
    process = subprocess.Popen(
        command,
        cwd="./musicgen-dreamboothing", # <-- Clave: ejecutar desde aquí
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding='utf-8'
    )
    
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
                gr.Markdown("### Paso 1: Preparar Dataset (Opcional)")
                gr.Markdown("Si aún no tienes un `metadata.jsonl`, pon la ruta a tu carpeta de audios y haz clic aquí.")
                prep_dataset_path_input = gr.Textbox(label="Ruta a la Carpeta de tu Dataset", placeholder="/mnt/datos/ModeloAudio/training_data")
                run_prep_button = gr.Button("🤖 Generar `metadata.jsonl`")
                prep_output = gr.Textbox(label="Registro del Proceso de Preparación", interactive=False, lines=5)

            with gr.Group():
                gr.Markdown("### Paso 2: Configurar y Lanzar Entrenamiento")
                output_dir_input = gr.Textbox(label="Nombre de la Carpeta para el LoRA", value="./mi_lora_final")
                with gr.Row():
                    epochs_input = gr.Slider(label="Épocas (Epochs)", minimum=1, maximum=100, value=15, step=1)
                    lr_input = gr.Number(label="Tasa de Aprendizaje (Learning Rate)", value=0.0001)
                    max_duration_input = gr.Slider(label="Duración Máx. Audio (s)", minimum=10, maximum=300, value=180, step=1)
                with gr.Row():
                    lora_r_input = gr.Slider(label="Rango de LoRA (r)", minimum=4, maximum=128, value=32, step=4)
                    lora_alpha_input = gr.Slider(label="Alpha de LoRA", minimum=4, maximum=256, value=64, step=4)

                launch_train_button = gr.Button("🚀 ¡Lanzar Entrenamiento!", variant="primary")
                train_output = gr.Textbox(label="Registro del Entrenamiento", interactive=False, lines=15)

        with gr.TabItem("🎵 Generador de Música (Inferencia)"):
            with gr.Row():
                with gr.Column(scale=2):
                    prompt_input = gr.Textbox(label="Prompt Musical", placeholder="Ej: Un solo de piano clásico, melancólico y lento")
                    duration_input = gr.Slider(minimum=5, maximum=60, value=15, label="Duración (segundos)")
                    lora_path_input = gr.Textbox(label="Ruta al Adaptador LoRA", placeholder="./musicgen-dreamboothing/mi_lora_final")
                with gr.Column(scale=1):
                    generate_button = gr.Button("🎹 Generar Música", variant="primary")
                    status_output = gr.Textbox(label="Estado del Modelo", interactive=False, value="ℹ️ Usando modelo base.")
            audio_output = gr.Audio(label="Resultado Generado")

    # --- Conexiones de la Interfaz ---
    run_prep_button.click(
        fn=run_prepare_dataset,
        inputs=[prep_dataset_path_input],
        outputs=[prep_output]
    )
    launch_train_button.click(
        fn=modify_and_run_training,
        inputs=[prep_dataset_path_input, output_dir_input, epochs_input, lr_input, lora_r_input, lora_alpha_input, max_duration_input],
        outputs=[train_output]
    )
    generate_button.click(
        fn=generate_music,
        inputs=[prompt_input, duration_input, lora_path_input],
        outputs=[audio_output]
    )
    lora_path_input.submit(
        fn=load_lora_model,
        inputs=[lora_path_input],
        outputs=[status_output]
    )

if __name__ == "__main__":
    demo.launch(share=False)