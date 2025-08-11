# Gemma-3 4B Fine-tuning with Unsloth

This repository contains a Jupyter notebook for fine-tuning Google's Gemma-3 4B model using the Unsloth library. The notebook demonstrates how to efficiently fine-tune the model using LoRA (Low-Rank Adaptation) on a high-quality instruction dataset.

## 📋 Overview

The notebook implements a complete fine-tuning pipeline for Gemma-3 4B including:
- Model loading with 4-bit quantization
- LoRA adapter configuration
- Dataset preparation with proper chat templating
- Training with response masking
- Model evaluation and inference
- Multiple saving formats (LoRA adapters, merged model, GGUF)

## 🚀 Features

- **Memory Efficient**: Uses 4-bit quantization and LoRA adapters
- **Fast Training**: Leverages Unsloth's optimizations for 2x faster training
- **High Quality Dataset**: Trained on FineTome-100k dataset with 100,000 high-quality instruction examples
- **Proper Chat Templating**: Uses Gemma-3 chat format for consistent conversation handling
- **Response-Only Training**: Trains only on model responses, ignoring user inputs
- **Multiple Export Options**: Supports LoRA, merged model, and GGUF formats

## 🛠️ Setup

### Requirements

The notebook installs all necessary dependencies automatically:

```bash
pip install unsloth
pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
```

### Hardware Requirements

- **GPU**: Tesla T4 or better (tested on Tesla T4 with 14.7GB memory)
- **Memory**: ~7.4GB GPU memory for training
- **Training Time**: ~7 minutes for 32 steps

## 📂 Dataset

The notebook uses the [FineTome-100k](https://huggingface.co/datasets/mlabonne/FineTome-100k) dataset:
- **Size**: 100,000 high-quality instruction examples
- **Format**: ShareGPT conversation format
- **Quality**: Curated dataset with diverse instruction-following examples

## 🔧 Configuration

### Model Configuration
```python
model, tokenizer = FastModel.from_pretrained(
    model_name="unsloth/gemma-3-4b-it",
    max_seq_length=2048,
    load_in_4bit=True,
    full_finetuning=False,
)
```

### LoRA Configuration
```python
model = FastModel.get_peft_model(
    model,
    r=8,                           # LoRA rank
    lora_alpha=8,                  # LoRA alpha
    lora_dropout=0,                # LoRA dropout
    bias="none",                   # Bias configuration
    finetuning_vision_layers=False, # Text-only fine-tuning
    finetuning_language_layers=True,
    finetuning_attention_modules=True,
    finetuning_mlp_modules=True,
)
```

### Training Configuration
```python
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=SFTConfig(
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        max_steps=32,
        learning_rate=2e-4,
        logging_steps=1,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
    )
)
```

## 🎯 Training Process

1. **Model Loading**: Load Gemma-3 4B with 4-bit quantization
2. **LoRA Setup**: Add LoRA adapters for efficient fine-tuning
3. **Data Preparation**: Format dataset using Gemma-3 chat template
4. **Response Masking**: Configure training to only learn from model responses
5. **Training**: Fine-tune for 32 steps with optimized settings
6. **Evaluation**: Test model with sample prompts

### Training Results
- **Trainable Parameters**: 19,248,896 out of 4,319,328,368 (0.45%)
- **Training Time**: ~7 minutes for 32 steps
- **Memory Usage**: ~7.4GB peak GPU memory
- **Final Loss**: ~0.82

## 💾 Saving Options

### 1. LoRA Adapters Only
```python
model.save_pretrained("gemma-3-lora")
tokenizer.save_pretrained("gemma-3-lora")
```

### 2. Merged Model (Float16)
```python
model.save_pretrained_merged("gemma-3-finetune", tokenizer)
```

### 3. GGUF Format (for llama.cpp)
```python
model.save_pretrained_gguf(
    "gemma-3-finetune",
    quantization_type="Q8_0"
)
```

## 🔍 Usage

### Loading the Fine-tuned Model
```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="path/to/saved/model",
    max_seq_length=2048,
    load_in_4bit=True,
)
```

### Inference
```python
messages = [{
    "role": "user",
    "content": [{"type": "text", "text": "Your question here"}]
}]

text = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
)

outputs = model.generate(
    **tokenizer([text], return_tensors="pt").to("cuda"),
    max_new_tokens=64,
    temperature=1.0,
    top_p=0.95,
    top_k=64,
)
```

## 📊 Performance

- **Memory Efficiency**: Only trains 0.45% of model parameters
- **Speed**: 2x faster training with Unsloth optimizations
- **Quality**: Maintains high-quality outputs while being efficient
- **Compatibility**: Works with standard Hugging Face ecosystem

## 🔗 Key Dependencies

- **unsloth**: Fast fine-tuning library
- **transformers**: Hugging Face transformers
- **trl**: Training with reinforcement learning
- **datasets**: Dataset loading and processing
- **peft**: Parameter-efficient fine-tuning
- **bitsandbytes**: 4-bit quantization support

## 📝 Notes

- The notebook is designed to run on Google Colab with a free Tesla T4 GPU
- Training is optimized for speed with 32 steps - increase for production use
- The model uses Gemma-3 chat format: `user\nQuestion\nmodel\nAnswer`
- Response-only training improves instruction-following accuracy

## 🤝 Contributing

Feel free to submit issues, improvements, or suggestions for the fine-tuning process.

## 📄 License

This project follows the licensing terms of the Unsloth library and Gemma-3 model.

## 🙏 Acknowledgments

- [Unsloth](https://github.com/unslothai/unsloth) for the efficient fine-tuning library
- [Google](https://ai.google.dev/gemma) for the Gemma-3 model
- [Maxime Labonne](https://huggingface.co/datasets/mlabonne/FineTome-100k) for the FineTome-100k dataset