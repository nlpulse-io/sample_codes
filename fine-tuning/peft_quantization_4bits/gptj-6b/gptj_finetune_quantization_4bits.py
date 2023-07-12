# Baseado em: https://towardsdatascience.com/qlora-fine-tune-a-large-language-model-on-your-gpu-27bed5a03e2b


import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, Trainer, TrainingArguments, DataCollatorForLanguageModeling

from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model

from datasets import load_dataset

'''
https://huggingface.co/EleutherAI/gpt-j-6b#out-of-scope-use
GPT-J-6B was trained on an English-language only dataset, and is thus not suitable for translation or generating text in other languages.
GPT-J-6B has not been fine-tuned for downstream contexts in which language models are commonly deployed, such as writing genre prose, or commercial chatbots. This means GPT-J-6B will not respond to a given prompt the way a product like ChatGPT does. This is because, unlike this model, ChatGPT was fine-tuned using methods such as Reinforcement Learning from Human Feedback (RLHF) to better “follow” human instructions.

'''



# model_name = "EleutherAI/gpt-neox-20b"
model_name = "EleutherAI/gpt-j-6b" # pytorch_model.bin => 24GB // Treino => 7.5GB de GPU
model_name = os.environ.get("model_name", model_name)

train_output_dir = "gptj_train_outputs"
train_output_dir = os.environ.get("train_output_dir", train_output_dir)

model_output_dir = "gptj_finetuned_model"
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
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant_config, device_map={"":0})
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)
lora_args = LoraConfig(
    r=8, 
    lora_alpha=32, 
    # target_modules=["query_key_value"], # gpt-neox-20b
    target_modules=["q_proj", "v_proj"], # gpt-j-6b
    lora_dropout=0.05, 
    bias="none", 
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_args)
model.print_trainable_parameters()

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
CUDA_VISIBLE_DEVICES=1 model_output_dir=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits train_output_dir=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits_outputs python gptj_finetune_quantization_4bits.py

# Saida (20 steps):
trainable params: 3,670,016 || all params: 3,235,980,512 || trainable%: 0.11341279672082277
{'loss': 1.9432, 'learning_rate': 0.0001, 'epoch': 0.0}
{'loss': 2.6189, 'learning_rate': 0.0002, 'epoch': 0.01}
{'loss': 1.9313, 'learning_rate': 0.00018888888888888888, 'epoch': 0.01}
{'loss': 2.1943, 'learning_rate': 0.00017777777777777779, 'epoch': 0.01}
{'loss': 1.8241, 'learning_rate': 0.0001666666666666667, 'epoch': 0.02}
{'loss': 1.0868, 'learning_rate': 0.00015555555555555556, 'epoch': 0.02}
{'loss': 1.7539, 'learning_rate': 0.00014444444444444444, 'epoch': 0.02}
{'loss': 2.2778, 'learning_rate': 0.00013333333333333334, 'epoch': 0.03}
{'loss': 2.467, 'learning_rate': 0.00012222222222222224, 'epoch': 0.03}
{'loss': 2.1245, 'learning_rate': 0.00011111111111111112, 'epoch': 0.03}
{'loss': 1.615, 'learning_rate': 0.0001, 'epoch': 0.04}
{'loss': 2.151, 'learning_rate': 8.888888888888889e-05, 'epoch': 0.04}
{'loss': 2.1599, 'learning_rate': 7.777777777777778e-05, 'epoch': 0.04}
{'loss': 2.1273, 'learning_rate': 6.666666666666667e-05, 'epoch': 0.04}
{'loss': 2.2875, 'learning_rate': 5.555555555555556e-05, 'epoch': 0.05}
{'loss': 2.1918, 'learning_rate': 4.4444444444444447e-05, 'epoch': 0.05}
{'loss': 1.8144, 'learning_rate': 3.3333333333333335e-05, 'epoch': 0.05}
{'loss': 2.7467, 'learning_rate': 2.2222222222222223e-05, 'epoch': 0.06}
{'loss': 1.1798, 'learning_rate': 1.1111111111111112e-05, 'epoch': 0.06}
{'loss': 1.8061, 'learning_rate': 0.0, 'epoch': 0.06}
{'train_runtime': 127.7297, 'train_samples_per_second': 1.253, 'train_steps_per_second': 0.157, 'train_loss': 2.015064871311188, 'epoch': 0.06}

# Saida (939 steps - duracao 1h45min):
...
{'loss': 1.1051, 'learning_rate': 1.28068303094984e-06, 'epoch': 2.98}
{'loss': 1.425, 'learning_rate': 1.0672358591248667e-06, 'epoch': 2.98}
{'loss': 1.0258, 'learning_rate': 8.537886872998934e-07, 'epoch': 2.99}
{'loss': 1.3734, 'learning_rate': 6.4034151547492e-07, 'epoch': 2.99}
{'loss': 1.3357, 'learning_rate': 4.268943436499467e-07, 'epoch': 2.99}
{'loss': 1.1659, 'learning_rate': 2.1344717182497335e-07, 'epoch': 3.0}
{'train_runtime': 6247.8916, 'train_samples_per_second': 1.204, 'train_steps_per_second': 0.15, 'train_loss': 1.727637820223523, 'epoch': 3.0}

# Arquivos gerados:
$ ls -lht /var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits
total 15M
-rw-rw-r-- 1 dockeradmin dockeradmin 417 Jul 10 09:30 adapter_config.json
-rw-rw-r-- 1 dockeradmin dockeradmin 15M Jul 10 09:30 adapter_model.bin
-rw-rw-r-- 1 dockeradmin dockeradmin 440 Jul 10 09:30 README.md

# Consumo:
$ watch -n 3 'nvidia-smi && free -h'
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|100%   89C    P2   166W / 170W |   7439MiB / 12288MiB |     93%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
               total        used        free      shared  buff/cache   available
Mem:            77Gi        14Gi        23Gi        79Mi        39Gi        62Gi
Swap:           37Gi          0B        37Gi



# Preparacao:
cd /var/server1/docker/volumes/code/project/lab1/poc_qaml2/experimentos/peft_quantization
git reset --hard && git fetch && git pull
python3 -m venv ~/venv/peft_quantization
source ~/venv/peft_quantization/bin/activate
pip install --upgrade pip
pip install -q -U bitsandbytes
pip install -q -U git+https://github.com/huggingface/transformers.git 
pip install -q -U git+https://github.com/huggingface/peft.git
# pip install -q -U git+https://github.com/huggingface/accelerate.git
# current version of Accelerate on GitHub breaks QLoRa
# Using standard pip instead
pip install -q -U accelerate
pip install -q -U datasets
pip install -q -U scipy

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