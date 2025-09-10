import gradio as gr
import torch
from transformers import AutoProcessor, MusicgenForConditionalGeneration
import scipy.io.wavfile

# --- Configuración Inicial ---
# Comprobamos si hay una GPU compatible con CUDA disponible
if not torch.cuda.is_available():
    raise RuntimeError("Este script requiere una GPU con CUDA para usar Flash Attention 2.")

DEVICE = "cuda"
DTYPE = torch.float16

# --- Carga del Modelo y Procesador ---
# Cargamos el procesador y el modelo desde Hugging Face.
# La clave aquí es 'attn_implementation="flash_attention_2"', que activa la optimización.
# Usamos float16 para acelerar la inferencia y reducir el consumo de memoria.
print("Cargando el procesador de MusicGen...")
processor = AutoProcessor.from_pretrained("facebook/musicgen-large")

print("Cargando el modelo Musicgen-large con Flash Attention 2...")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-large",
    torch_dtype=DTYPE,
    attn_implementation="flash_attention_2"
).to(DEVICE)

print("Modelo cargado y listo en el dispositivo:", DEVICE)

def generate_music(prompt, duration_seconds):
    """
    Función principal que genera música a partir de un prompt de texto.

    Args:
        prompt (str): La descripción de la música a generar.
        duration_seconds (int): La duración deseada del audio en segundos.

    Returns:
        tuple: Una tupla con la frecuencia de muestreo y el audio en formato numpy array.
    """
    if not prompt:
        raise gr.Error("El prompt de texto no puede estar vacío. ¡Describe la música que quieres crear!")

    print(f"Generando música para el prompt: '{prompt}' con una duración de {duration_seconds} segundos.")

    # El modelo MusicGen genera aproximadamente 50 tokens por segundo de audio.
    # Calculamos el número máximo de tokens nuevos a generar a partir de la duración.
    max_new_tokens = int(duration_seconds * 50)

    # Preparamos las entradas para el modelo
    inputs = processor(
        text=[prompt],
        padding=True,
        return_tensors="pt"
    ).to(DEVICE)

    # Configuramos los parámetros de generación
    generation_kwargs = {
        "do_sample": True,
        "guidance_scale": 3,
        "max_new_tokens": max_new_tokens,
        "top_k": 250,
    }

    # Generamos la forma de onda del audio
    with torch.no_grad():
        audio_values = model.generate(**inputs, **generation_kwargs)[0]

    # La salida del modelo es un tensor de PyTorch, lo movemos a la CPU y lo convertimos a un array de numpy
    audio_numpy = audio_values.cpu().numpy()
    sampling_rate = model.config.audio_encoder.sampling_rate

    print("Generación completada.")
    return (sampling_rate, audio_numpy)

# --- Creación de la Interfaz de Gradio ---
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🎵 MusicGen con Flash Attention 2
        ### Genera música de alta calidad a partir de descripciones de texto.
        Este demo utiliza el modelo `musicgen-large` de Meta AI, optimizado con **Flash Attention 2**
        para una inferencia más rápida y con menor consumo de memoria en GPUs compatibles.
        """
    )
    with gr.Row():
        with gr.Column():
            prompt_input = gr.Textbox(
                label="Descripción de la música (Prompt)",
                placeholder="Ej: Una melodía de piano clásica y melancólica, con violines de fondo, tempo lento.",
                lines=3
            )
            duration_slider = gr.Slider(
                minimum=5,
                maximum=60,
                value=10,
                step=1,
                label="Duración (segundos)"
            )
            generate_button = gr.Button("🎹 Generar Música", variant="primary")

        with gr.Column():
            audio_output = gr.Audio(label="Resultado", type="numpy")

    generate_button.click(
        fn=generate_music,
        inputs=[prompt_input, duration_slider],
        outputs=audio_output
    )

    gr.Markdown("---")
    gr.Markdown("### Ejemplos para probar:")
    gr.Examples(
        examples=[
            ["Un riff de guitarra de rock de los 80 con una batería potente y contundente.", 12],
            ["Música Lo-fi hip hop relajante, ideal para estudiar, con un ritmo suave de batería y un piano eléctrico.", 15],
            ["Una pieza orquestal épica y cinematográfica con coros dramáticos y percusión atronadora.", 20],
            ["Reggae acústico con una línea de bajo prominente y un ritmo relajado.", 10],
        ],
        inputs=[prompt_input, duration_slider]
    )


if __name__ == "__main__":
    demo.launch(debug=True)

