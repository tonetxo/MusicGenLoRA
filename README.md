# MusicGen LoRA Tuner & Generator

This repository provides a comprehensive Gradio-based user interface for fine-tuning the `facebook/musicgen-small` model using LoRA (Low-Rank Adaptation) and generating music with the resulting custom models. It streamlines the entire workflow, from dataset preparation and augmentation to training and inference, all within a single, user-friendly application.

## ✨ Features

*   **Gradio Interface**: A clean, tab-based UI for managing all tasks.
*   **DreamBooth/LoRA Training**: Fine-tune MusicGen on your own audio datasets to create specialized LoRA models.
*   **Automated Data Preparation**: Includes a tool to automatically generate captions and metadata (`metadata.jsonl`) from a folder of audio files using an integrated audio-tagging model.
*   **Data Augmentation**: Simple yet effective audio augmentation (pitch shifting, time stretching, volume adjustment) to expand your training dataset.
*   **Ollama Integration**: Enhance your creative prompts by leveraging local LLMs (via Ollama) to generate more descriptive and effective text for music generation.
*   **Prompt Management**: Save, load, and manage your favorite prompts directly within the UI.
*   **Dynamic LoRA Switching**: Easily load and switch between different LoRA models during inference without restarting the application.
*   **Flash Attention 2**: Optimized for speed and memory efficiency during training and inference.
*   **Configuration Persistence**: All your settings (paths, training parameters, inference settings) are automatically saved to a `settings.json` file.

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```

2.  **Install Python dependencies:**
    It is highly recommended to use a virtual environment (e.g., venv or conda). The project requires a PyTorch version compatible with your CUDA drivers.

    ```bash
    pip install -r requirements.txt
    ```

3.  **(Optional) Install Ollama:**
    For using the prompt enhancement feature, you need to have [Ollama](https://ollama.com/) installed and running. You should also pull a model to use, for example:
    ```bash
    ollama pull llama3
    ```

## 🚀 Usage

Launch the application by running:

```bash
python app.py
```

The interface is organized into several tabs, each corresponding to a specific part of the workflow.

### 1. 🛠️ Train LoRA

This is the main tab for data preparation and training.

#### Data Preparation

1.  **Place your audio files** (e.g., `.wav`, `.mp3`) in a single folder.
2.  In the UI, enter the path to this folder in the **"Ruta a la carpeta con tus audios"** textbox.
3.  Click **"Generar `metadata.jsonl`"**. This will analyze each audio file, generate a descriptive caption, and create the `metadata.jsonl` file required for training.

#### Data Augmentation (Optional)

1.  After generating the `metadata.jsonl`, you can augment your dataset.
2.  Specify an **"Ruta de salida para dataset augmentado"**.
3.  Click **"Augmentar Dataset"**. This will create a new folder with the original and augmented audio files, along with a new `metadata.jsonl` file.
4.  To use this augmented data for training, check the **"Usar dataset augmentado para entrenamiento"** box.

#### Training

1.  **Configure the training parameters**:
    *   **Carpeta de salida (LoRA)**: Where the final LoRA model will be saved.
    *   **Épocas**: Number of training epochs.
    *   **Learning Rate**: The learning rate for the optimizer.
    *   **Duración máx. del audio (s)**: Maximum audio duration to use for training samples.
    *   **R (rank)** and **Alpha**: Key parameters for the LoRA configuration.
    *   **Semilla (entrenamiento)**: A seed for reproducibility.
2.  Click **"Lanzar entrenamiento"**. The training progress will be displayed in the log window.

### 2. ✍️ Gestor de Prompts

This tab helps you manage and enhance your prompts.

*   **Save/Load**: Save your frequently used prompts with a name, and quickly load them from the dropdown menu.
*   **Enhance with Ollama**:
    1.  Select an available Ollama model.
    2.  Write a basic idea in the prompt text area.
    3.  Click **"Mejorar con Ollama"** to get a more detailed, English-language prompt suitable for MusicGen.
    4.  You can optionally check **"Usar captions del dataset como contexto"** to guide the LLM with keywords from your training data.
*   Click **"Usar este Prompt en el Generador"** to send the current prompt to the inference tab.

### 3. 🎵 Generador (Inferencia)

This tab is for generating music.

1.  **Write your prompt** or send one from the Prompt Manager.
2.  **(Optional) Load a LoRA model**: Drag and drop the `adapter_config.json` and `adapter_model.safetensors` files from your LoRA output directory onto the file input area. The "Modelo activo" status will update to "✅ LoRA activo". To unload the LoRA and revert to the base model, simply clear the file input.
3.  **Set generation parameters**:
    *   **Duración (s)**: Length of the generated audio.
    *   **Semilla (-1 = aleatoria)**: Seed for reproducibility. A value of -1 uses a random seed.
    *   **Advanced Settings**: Adjust `Guidance Scale (CFG)`, `Temperatura`, `Top-k`, and `Top-p` to control the creativity and coherence of the output.
4.  Click **"Generar"**. The generated audio will appear in the audio player below.
5.  You can save the generated audio to disk using the **"Guardar Audio Generado"** button.

## ⚙️ Configuration

The application uses a `settings.json` file to store your preferences for paths, training parameters, and inference settings. This file is loaded at startup and saved automatically whenever you change a setting in the UI, ensuring your configuration is preserved across sessions.

## 📦 Dependencies

The project relies on the following key libraries:

*   `torch` & `torchaudio`: For deep learning and audio processing.
*   `transformers`, `accelerate`, `peft`: The Hugging Face ecosystem for models, training, and PEFT (Parameter-Efficient Fine-Tuning).
*   `gradio`: For creating the web-based user interface.
*   `librosa`: For audio analysis (used in data preparation).
*   `requests`: For communicating with the Ollama API.