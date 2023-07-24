# Baseado em: https://medium.com/@jain.sm/finetuning-llama-2-0-on-colab-with-1-gpu-7ea73a8d3db9


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from datasets import load_dataset

'''

'''


model_name = "meta-llama/Llama-2-7b-chat-hf"
model_name = os.environ.get("model_name", model_name)

train_output_dir = "llama2chat_train_outputs"
train_output_dir = os.environ.get("train_output_dir", train_output_dir)

model_output_dir = "llama2chat_finetuned_model"
model_output_dir = os.environ.get("model_output_dir", model_output_dir)

max_steps = -1
max_steps = os.environ.get("max_steps", str(max_steps))
max_steps = int(max_steps)

num_train_epochs = 3
num_train_epochs = os.environ.get("num_train_epochs", str(num_train_epochs))
num_train_epochs = int(num_train_epochs)

gradient_accumulation_steps = 8
gradient_accumulation_steps = os.environ.get("gradient_accumulation_steps", str(gradient_accumulation_steps))
gradient_accumulation_steps = int(gradient_accumulation_steps)


# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, 
        use_auth_token=True)
tokenizer.pad_token = tokenizer.eos_token

# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"":0}, 
        use_auth_token=True)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
lora_args = LoraConfig(
    r=8, 
    lora_alpha=32, 
    # target_modules=["query_key_value"], # gpt-neox-20b
    target_modules=["q_proj", "v_proj"], # gpt-j-6b llama2chat
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_args)

model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 3,504,607,232 || trainable%: 0.11967971650867153


# dataset
data = load_dataset("Abirate/english_quotes")
data = data.map(lambda samples: tokenizer(samples["quote"]), batched=True)


# treino
trainer = Trainer(
    model=model,
    train_dataset=data["train"],
    args=TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=gradient_accumulation_steps, # 8
        warmup_steps=2,
        max_steps=max_steps, # 20 => 20/939
        num_train_epochs=num_train_epochs, # 3
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,
        output_dir=train_output_dir,
        optim="paged_adamw_8bit"
    ),
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
trainer.train()

# Saves the Trainer state, since Trainer.save_model saves only the tokenizer with the model
trainer.save_state()
model.save_pretrained(model_output_dir)


'''

# Execucao:
cd /var/server1/docker/volumes/code/project/lab1/poc_qaml2/experimentos/peft_quantization
source ~/venv/peft_quantization/bin/activate
CUDA_VISIBLE_DEVICES=1 gradient_accumulation_steps=8 model_output_dir=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_8gradsteps train_output_dir=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_outputs_8gradsteps python llama2chat_finetune_quantization_4bits.py

# Saida (gradient_accumulation_steps=8 - 939 steps - duracao 1h54min):
{'loss': 0.72, 'learning_rate': 2.1344717182497334e-06, 'epoch': 2.98}
{'loss': 0.7017, 'learning_rate': 1.92102454642476e-06, 'epoch': 2.98}
{'loss': 0.7088, 'learning_rate': 1.7075773745997868e-06, 'epoch': 2.98}
{'loss': 0.8118, 'learning_rate': 1.4941302027748133e-06, 'epoch': 2.99}
{'loss': 0.8813, 'learning_rate': 1.28068303094984e-06, 'epoch': 2.99}
{'loss': 0.836, 'learning_rate': 1.0672358591248667e-06, 'epoch': 2.99}
{'loss': 0.6297, 'learning_rate': 8.537886872998934e-07, 'epoch': 3.0}
{'train_runtime': 6865.9735, 'train_samples_per_second': 1.096, 'train_steps_per_second': 0.137, 'train_loss': 1.1723256691322286, 'epoch': 3.0}


# Arquivos gerados:
$ ls -lht /var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits
total 17M
-rw-rw-r-- 1 dockeradmin dockeradmin 427 Jul 24 11:42 adapter_config.json
-rw-rw-r-- 1 dockeradmin dockeradmin 17M Jul 24 11:42 adapter_model.bin
-rw-rw-r-- 1 dockeradmin dockeradmin 440 Jul 24 11:42 README.md

$ ls -lht /var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_8gradsteps
total 17M
-rw-rw-r-- 1 dockeradmin dockeradmin 427 Jul 24 14:00 adapter_config.json
-rw-rw-r-- 1 dockeradmin dockeradmin 17M Jul 24 14:00 adapter_model.bin
-rw-rw-r-- 1 dockeradmin dockeradmin 440 Jul 24 14:00 README.md


# Consumo:
$ nvidia-smi && free -h
Mon Jul 24 09:56:03 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|100%   87C    P2   168W / 170W |   6854MiB / 12288MiB |     98%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
               total        used        free      shared  buff/cache   available
Mem:            77Gi        13Gi       1.1Gi       116Mi        63Gi        63Gi
Swap:           37Gi       3.8Gi        34Gi



# Preparacao:
cd /var/server1/docker/volumes/code/project/lab1/poc_qaml2/experimentos/peft_quantization
git reset --hard && git fetch && git pull
python3 -m venv ~/venv/peft_quantization
source ~/venv/peft_quantization/bin/activate
pip install --upgrade pip
pip install -U bitsandbytes
pip install -U git+https://github.com/huggingface/transformers.git 
pip install -U git+https://github.com/huggingface/peft.git
# pip install -U git+https://github.com/huggingface/accelerate.git
# current version of Accelerate on GitHub breaks QLoRa
# Using standard pip instead
pip install -U accelerate
pip install -U datasets
pip install -U scipy

$ python -V
Python 3.10.6
$ pip list
Package                  Version
------------------------ -----------
accelerate               0.20.3
aiohttp                  3.8.4
aiosignal                1.3.1
async-timeout            4.0.2
attrs                    23.1.0
bitsandbytes             0.39.1
certifi                  2023.5.7
charset-normalizer       3.2.0
cmake                    3.26.4
datasets                 2.13.1
dill                     0.3.6
filelock                 3.12.2
frozenlist               1.3.3
fsspec                   2023.6.0
huggingface-hub          0.16.4
idna                     3.4
Jinja2                   3.1.2
lit                      16.0.6
MarkupSafe               2.1.3
mpmath                   1.3.0
multidict                6.0.4
multiprocess             0.70.14
networkx                 3.1
numpy                    1.25.1
nvidia-cublas-cu11       11.10.3.66
nvidia-cuda-cupti-cu11   11.7.101
nvidia-cuda-nvrtc-cu11   11.7.99
nvidia-cuda-runtime-cu11 11.7.99
nvidia-cudnn-cu11        8.5.0.96
nvidia-cufft-cu11        10.9.0.58
nvidia-curand-cu11       10.2.10.91
nvidia-cusolver-cu11     11.4.0.1
nvidia-cusparse-cu11     11.7.4.91
nvidia-nccl-cu11         2.14.3
nvidia-nvtx-cu11         11.7.91
packaging                23.1
pandas                   2.0.3
peft                     0.4.0.dev0
pip                      23.1.2
psutil                   5.9.5
pyarrow                  12.0.1
python-dateutil          2.8.2
pytz                     2023.3
PyYAML                   6.0
regex                    2023.6.3
requests                 2.31.0
safetensors              0.3.1
scipy                    1.11.1
setuptools               59.6.0
six                      1.16.0
sympy                    1.12
tokenizers               0.13.3
torch                    2.0.1
tqdm                     4.65.0
transformers             4.31.0.dev0
triton                   2.0.0
typing_extensions        4.7.1
tzdata                   2023.3
urllib3                  2.0.3
wheel                    0.40.0
xxhash                   3.2.0
yarl                     1.9.2


https://github.com/huggingface/peft/blob/39ef2546d5d9b8f5f8a7016ec10657887a867041/src/peft/utils/other.py#L220
TRANSFORMERS_MODELS_TO_LORA_TARGET_MODULES_MAPPING = {
    "t5": ["q", "v"],
    "mt5": ["q", "v"],
    "bart": ["q_proj", "v_proj"],
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "blip-2": ["q", "v", "q_proj", "v_proj"],
    "opt": ["q_proj", "v_proj"],
    "gptj": ["q_proj", "v_proj"],
    "gpt_neox": ["query_key_value"],
    "gpt_neo": ["q_proj", "v_proj"],
    "bert": ["query", "value"],
    "roberta": ["query", "value"],
    "xlm-roberta": ["query", "value"],
    "electra": ["query", "value"],
    "deberta-v2": ["query_proj", "value_proj"],
    "deberta": ["in_proj"],
    "layoutlm": ["query", "value"],
    "llama": ["q_proj", "v_proj"],
    "chatglm": ["query_key_value"],
    "gpt_bigcode": ["c_attn"],
    "mpt": ["Wqkv"],
}


'''