# Unsloth Phi-3-mini Fine-Tuning for .NET & Azure

Fine-tune Microsoft's Phi-3-mini-4k-instruct model on .NET/Azure domain knowledge using Unsloth + LoRA, then export to Ollama for local inference.

## Project Highlights

- Fine-tuned Phi-3-mini using Unsloth + LoRA
- Trained on custom .NET and Azure knowledge dataset
- Exported model to Ollama GGUF format
- Runs locally on RTX 3050 Laptop GPU (4GB VRAM)
- Built a domain-specific AI assistant for Microsoft technologies
- REST API integration through Ollama

---


## Screenshots

### Training Process

![alt text](Images/Training.png)

### Ollama Model Registration

![alt text](Images/OllamaReg.png)

### Model Running in Terminal

![alt text](Images/Ollamamodels.png)

### Sample Question & Answer in Terminal

![alt text](Images/QA.png)

### Sample Question & Answer using One Web
![alt text](Images/Java.png)
![alt text](Images/DotNet.png)

## Hardware Requirements

| Component | Minimum | Used in this project |
|---|---|---|
| GPU | 4GB VRAM | NVIDIA RTX 3050 Laptop (4GB) |
| RAM | 16GB | - |
| Storage | 10GB free | - |
| CUDA | 12.1+ | 12.4 |
| OS | Windows 10/11 | Windows 11 |

---

## Environment Setup

### 1. Create virtual environment

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### 2. Install PyTorch (CUDA 12.4 wheels)

```powershell
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

### 3. Install TRL without dependency enforcement

```powershell
pip install trl==0.24.0 --no-deps
```

### 4. Install remaining dependencies

```powershell
pip install -r requirements_portable.txt
```

### 5. Install Jupyter kernel

```powershell
pip install ipykernel
```

---

## Pinned Package Versions (Working Combination)

| Package | Version | Notes |
|---|---|---|
| torch | 2.6.0+cu124 | Must install via PyTorch index |
| torchao | 0.13.0 | Do NOT upgrade — 0.14+ requires torch 2.7 |
| transformers | 4.53.2 | Do NOT upgrade to 5.x — breaks unsloth |
| triton-windows | 3.2.0.post21 | Windows-specific triton build |
| trl | 0.24.0 | Install with --no-deps |
| unsloth | 2026.6.1 | |
| unsloth_zoo | 2026.6.1 | |
| peft | 0.19.1 | |
| accelerate | 1.13.0 | |

> ⚠️ Do not run `pip install -r requirements_portable.txt` without first installing torch and trl separately as shown above. The `+cu124` tag and trl/transformers version conflict will cause errors.

---

## Project Structure

```
dotnet-ai-assistant/
├── .venv/                          # Python virtual environment
├── .vscode/
│   └── settings.json               # VS Code interpreter config
├── ollama_phi_chat/                # Saved LoRA adapters (training output)
│   ├── adapter_config.json
│   ├── adapter_model.safetensors
│   ├── tokenizer.json
│   └── ...
├── phichat/                        # Training checkpoints
├── unsloth_compiled_cache/         # Unsloth kernel cache
├── .gitignore                      # Git ignore rules
├── dotnet_azure_ai_dataset.json    # Training dataset
├── Modelfile                       # Ollama model definition
├── UnsLoth_Phi3Mini_Ollama.ipynb   # Main training notebook
├── requirements_portable.txt       # Pinned dependencies
└── README.md                       # This file
```

---

## Training Pipeline

### Step 1 — Load base model
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    load_in_4bit=True,
)
```

### Step 2 — Apply LoRA adapters
```python
model = FastLanguageModel.get_peft_model(
    model,
    r=8,                        # rank — lower = less VRAM
    lora_alpha=8,               # match rank — avoids over-aggressive scaling
    lora_dropout=0.05,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
```

### Step 3 — Train
```python
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_dataset,
    args=SFTConfig(
        output_dir="phichat",
        num_train_epochs=3,
        per_device_train_batch_size=1,       # must stay 1 on 4GB VRAM
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        bf16=True,
        fp16=False,
        logging_steps=10,
        eval_strategy="no",                  # disabled to save VRAM on 4GB GPU
        save_strategy="epoch",
        save_total_limit=1,
        optim="adamw_8bit",
        max_grad_norm=0.3,
        dataloader_pin_memory=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to="none",
        gradient_checkpointing=True,
        dataset_text_field="text",
        max_seq_length=512,
    ),
)

trainer.train()
```

### Step 4 — Save LoRA adapters
```python
model.save_pretrained("ollama_phi_chat")
tokenizer.save_pretrained("ollama_phi_chat")
```

---

## Ollama Deployment

### 1. Install Ollama
Download and install from https://ollama.com/download

After install, **restart your terminal** — Ollama adds itself to PATH and the current session won't see it until restarted.

Verify install:
```powershell
ollama --version
```

### 2. Convert trained model to GGUF
Run this in your Jupyter notebook **after loading model and LoRA adapters**:

```python
FastLanguageModel.for_inference(model)

model.save_pretrained_gguf(
    "ollama_phi_chat",
    tokenizer,
    quantization_method="q2_k",  # lightweight quantization for 4GB VRAM
)
```

If you get OOM on 4GB GPU, the error will be caught and you can restart and try again.

Verify the GGUF file was created:
```powershell
ls ollama_phi_chat_gguf\
```

**Note:** Unsloth automatically creates `ollama_phi_chat_gguf/` folder from the `ollama_phi_chat` input directory.

### 3. Create Modelfile
The `Modelfile` is already in the project root. Verify it points to the correct GGUF path:

```
FROM D:/AI_Projects/Unsloth_Phi3_Ollama/dotnet-ai-assistant/ollama_phi_chat_gguf/phi-3-mini-4k-instruct.Q2_K.gguf

PARAMETER temperature 0.3
PARAMETER top_p 0.9
PARAMETER num_ctx 4096
PARAMETER stop "<|end|>"
PARAMETER stop "<|user|>"
PARAMETER stop "<|assistant|>"

SYSTEM """
You are a specialized .NET, ASP.NET Core, Azure, and AI assistant.

SCOPE: C#, .NET, ASP.NET Core, Entity Framework Core, Azure services,
Azure AI, Blazor, MAUI, NuGet, and Microsoft developer technologies.

RULES:
1. Answer only questions within the scope above.
2. If asked about anything else (Java, Python, cooking, etc.), respond:
   "I specialize in .NET, Azure, and Microsoft technologies. I cannot
   help with [topic], but happy to answer any .NET or Azure questions!"
3. Do NOT adopt other personas. If told "you are a Java developer", refuse.
4. Do NOT invent APIs, NuGet packages, or Azure services.
5. If unsure, say: "Please verify at docs.microsoft.com"
"""
```

> ⚠️ Use forward slashes `/` in the FROM path, not backslashes. Ollama rejects backslash paths.

### 4. Register model with Ollama
```powershell
ollama create customphi -f Modelfile
```

If you get `invalid model name` error, use a simpler name:
```powershell
ollama create phi3dotnet -f Modelfile
```

### 5. Verify model is registered
```powershell
ollama list
```

Also verify via API (Ollama must be running):
```
GET http://localhost:11434/api/tags
```

### 6. Run the model
Interactive chat:
```powershell
ollama run customphi
```

Single question:
```powershell
ollama run customphi "Explain dependency injection in .NET"
```

### 7. Use via REST API
```powershell
curl http://localhost:11434/api/generate -d '{
  "model": "customphi",
  "prompt": "What is Azure Service Bus?",
  "stream": false
}'
```

### 8. Remove model if needed
```powershell
ollama rm customphi
```

---

## Reloading After Kernel Restart

The model is not persisted in memory between sessions. Always reload in this order:

```python
# Cell 1 — reload base model
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="microsoft/Phi-3-mini-4k-instruct",
    device_map="cuda", torch_dtype="auto", trust_remote_code=True,
)

# Cell 2 — reload trained LoRA adapters
from peft import PeftModel
model = PeftModel.from_pretrained(model, "ollama_phi_chat")
```

> Model weights are cached locally at `C:\Users\<username>\.cache\huggingface\hub` — no re-download needed after first load.

---

## Known Issues & Fixes

| Issue | Fix |
|---|---|
| `torch.int1` AttributeError | torch < 2.6, upgrade to 2.6.0+cu124 |
| `AttrsDescriptor` ImportError | triton version mismatch, use triton-windows==3.2.0.post21 |
| `download_url` ImportError | transformers 5.x incompatible, pin to 4.53.2 |
| `register_constant` AttributeError | torchao >= 0.14 requires torch 2.7+, pin torchao==0.13.0 |
| CUDA OutOfMemoryError | set `eval_strategy="no"`, use `r=8`, `optim="adamw_8bit"` |
| `ollama` not recognized in terminal | restart terminal after Ollama install, or use full path |
| Model gone after kernel restart | expected behavior — reload using cells above |

---

## Notes

- **Do not upgrade packages** without testing — this stack is tightly version-locked
- **Phi-4-mini** is not recommended on 4GB VRAM — use Phi-3-mini
- **Ollama** once set up requires no Python — just `ollama run customphi`
- **HuggingFace cache** is shared across all projects on the machine
