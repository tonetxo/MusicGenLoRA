# train.py (Versión Final v5 - Usando BitsAndBytesConfig)
import argparse
import os
import torch
import traceback
import librosa

from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (
    AutoProcessor,
    MusicgenForConditionalGeneration,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig # <-- Importamos la configuración moderna
)

def main(args):
    # --- 1. Cargar el Procesador y el Modelo con QLoRA (Método Moderno) ---
    print("Iniciando entrenamiento: Cargando procesador y modelo base...")
    processor = AutoProcessor.from_pretrained(args.model_id)
    TARGET_SAMPLING_RATE = processor.feature_extractor.sampling_rate

    # ---- CONFIGURACIÓN DE CUANTIZACIÓN PRECISA (BitsAndBytesConfig) ----
    # Esta es la forma correcta y estable de cargar un modelo en 4-bit
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        # Excluimos de la cuantización las capas más sensibles para mantener la estabilidad
        llm_int8_skip_modules=["text_encoder.embed_tokens", "lm_head"],
    )

    model = MusicgenForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=bnb_config, # <-- Usamos la nueva configuración
        device_map="auto",
    )
    # -------------------------------------------------------------------

    # Mantenemos el parche de LayerNorms por si acaso (no hace daño)
    print("Estabilizando LayerNorms a float32...")
    for name, module in model.named_modules():
        if "layernorm" in name.lower() or "layer_norm" in name.lower():
            module.to(torch.float32)

    # --- 2. Configuración de LoRA ---
    print("Configurando el adaptador LoRA...")
    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()

    # --- 3. Preparación del Dataset ---
    print("Cargando y preparando el dataset...")
    dataset_file = os.path.join(args.dataset_path, "metadata.jsonl")
    dataset = load_dataset("json", data_files=dataset_file, split="train")

    def preprocess_function(examples):
        audio_paths = [os.path.join(args.dataset_path, x) for x in examples["file_name"]]
        raw_audios = []
        for path in audio_paths:
            audio, _ = librosa.load(path, sr=TARGET_SAMPLING_RATE, mono=True)
            raw_audios.append(audio)

        inputs = processor(
            audio=raw_audios,
            sampling_rate=TARGET_SAMPLING_RATE,
            text=examples["text"],
            padding="max_length",
            max_length=2048, 
            return_tensors="pt",
        )
        return inputs

    processed_dataset = dataset.map(
        preprocess_function,
        batched=True,
        batch_size=args.batch_size,
        num_proc=1,
        remove_columns=dataset.column_names,
    )

    # --- 4. Configuración del Entrenador ---
    print("Configurando los argumentos de entrenamiento...")
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=5,
        fp16=True,
        optim="paged_adamw_8bit",
        report_to="tensorboard",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=processed_dataset,
    )

    # --- 5. Iniciar Entrenamiento ---
    print("\n¡Todo listo! Iniciando el bucle de entrenamiento...")
    trainer.train()

    # --- 6. Guardar el Modelo ---
    print(f"Entrenamiento finalizado. Guardando el adaptador LoRA en: {args.output_dir}")
    model.save_pretrained(args.output_dir)

# --- Bloque de ejecución principal ---
if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Fine-tune MusicGen con LoRA")
        parser.add_argument("--model_id", type=str, default="facebook/musicgen-small", help="ID del modelo base de Hugging Face.")
        parser.add_argument("--dataset_path", type=str, required=True, help="Ruta a la carpeta con 'metadata.jsonl' y los archivos de audio.")
        parser.add_argument("--output_dir", type=str, required=True, help="Directorio donde se guardará el adaptador LoRA entrenado.")
        parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento.")
        parser.add_argument("--batch_size", type=int, default=1, help="Tamaño del lote. Para 6GB VRAM, debe ser 1.")
        parser.add_argument("--gradient_accumulation", type=int, default=4, help="Pasos de acumulación de gradiente.")
        parser.add_argument("--learning_rate", type=float, default=1e-4, help="Tasa de aprendizaje.")
        parser.add_argument("--lora_r", type=int, default=32, help="Rango (r) de LoRA.")
        parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha de LoRA.")
        parser.add_argument("--lora_dropout", type=float, default=0.1, help="Dropout de LoRA.")
        
        args = parser.parse_args()
        
        main(args)

    except Exception as e:
        print("\n!!!!!!!!!! ERROR INESPERADO ATRAPADO !!!!!!!!!!\n")
        traceback.print_exc()