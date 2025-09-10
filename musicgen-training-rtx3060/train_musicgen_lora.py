#!/usr/bin/env python3
"""
Script de entrenamiento optimizado para RTX 3060 6GB VRAM
"""
import torch
import torch.nn as nn
from transformers import AutoProcessor, MusicgenForConditionalGeneration, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType
from datasets import Dataset, Audio
import json
import os
import argparse
from pathlib import Path
import logging
import gc

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def setup_cuda_optimizations():
    """Configuraciones CUDA para RTX 3060."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_per_process_memory_fraction(0.85)
        logger.info(f"GPU: {torch.cuda.get_device_name()}")
        logger.info("Configuración optimizada para RTX 3060 6GB")

class MusicGenDataset:
    """Dataset optimizado para memoria limitada."""
    
    def __init__(self, dataset_path, max_duration=120, sample_rate=32000):
        self.dataset_path = Path(dataset_path)
        self.max_duration = max_duration
        self.sample_rate = sample_rate
        
        metadata_file = self.dataset_path / "metadata.jsonl"
        if not metadata_file.exists():
            raise FileNotFoundError(f"No se encontró {metadata_file}")
        
        self.data = []
        with open(metadata_file, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                if 'error' not in item:
                    # Filtrar por duración para conservar memoria
                    if item.get('duration', 0) <= self.max_duration:
                        self.data.append(item)
        
        logger.info(f"Dataset cargado: {len(self.data)} elementos válidos")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--max_duration", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    setup_cuda_optimizations()
    torch.manual_seed(args.seed)
    
    # Limpiar memoria
    torch.cuda.empty_cache()
    gc.collect()
    
    # Configuración LoRA conservadora
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    
    # Cargar modelo base
    model_id = "facebook/musicgen-small"
    processor = AutoProcessor.from_pretrained(model_id)
    model = MusicgenForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map={"": 0}
    )
    
    model = get_peft_model(model, lora_config)
    logger.info("Parámetros LoRA configurados para RTX 3060")
    
    # Configuración de entrenamiento conservadora
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=8,
        learning_rate=args.lr,
        warmup_steps=50,
        logging_steps=20,
        save_steps=300,
        save_strategy="steps",
        fp16=True,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        seed=args.seed,
        dataloader_pin_memory=False,  # Conservar memoria
    )
    
    logger.info("Entrenamiento configurado para RTX 3060 6GB VRAM")
    logger.info("Iniciando...")

if __name__ == "__main__":
    main()
