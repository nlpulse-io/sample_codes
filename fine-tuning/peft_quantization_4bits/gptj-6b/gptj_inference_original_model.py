
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig



# model_path = "EleutherAI/gpt-neox-20b"
model_path = "EleutherAI/gpt-j-6b" # pytorch_model.bin => 24GB
model_path = os.environ.get("model_path", model_path)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token


# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model
model = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant_config, device_map={"":0})


# inferencia
device = "cuda" # "cuda:0"
text_list = ["Ask not what your country", "Be the change that", "You only live once, but", "I'm selfish, impatient and"]
for text in text_list:
    inputs = tokenizer(text, return_tensors="pt").to(device)
    outputs = model.generate(**inputs, max_new_tokens=20)
    print('>> ', text, " => ", tokenizer.decode(outputs[0], skip_special_tokens=True))


'''
# Execucao:
cd /var/server1/docker/volumes/code/project/lab1/poc_qaml2/experimentos/peft_quantization
source ~/venv/peft_quantization/bin/activate
CUDA_VISIBLE_DEVICES=1 python gptj_inference_original_model.py
CUDA_VISIBLE_DEVICES=1 model_path=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits_applied python gptj_inference_original_model.py

# Frase original:
“I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.”
Marilyn Monroe

# Saida:
>>  I'm selfish, impatient and  =>  I'm selfish, impatient and a little bit of a control freak. I'm also a mom, a wife, a daughter,


# Consumo:
$ nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 515.105.01   Driver Version: 515.105.01   CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
| 37%   70C    P2   163W / 170W |   4923MiB / 12288MiB |     91%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

'''
