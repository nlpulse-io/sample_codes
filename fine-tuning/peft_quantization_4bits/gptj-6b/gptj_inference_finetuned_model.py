
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig, PeftModel


output_dir = "gptj_finetuned_model"
model_path = os.environ.get("model_path", output_dir)


# quantization config
quant_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

# model com adaptador PEFT LoRA
config = PeftConfig.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=quant_config, device_map={"":0})
model = PeftModel.from_pretrained(model, model_path)

# tokenizer
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token


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
CUDA_VISIBLE_DEVICES=1 model_path=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits python gptj_inference_finetuned_model.py

# Frase original:
“I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle. But if you can't handle me at my worst, then you sure as hell don't deserve me at my best.”
Marilyn Monroe

# Saida (todo os 939 steps e 3 epocas):
>>  I'm selfish, impatient and  =>  I'm selfish, impatient and a little insecure. I make mistakes, I am out of control and at times hard to handle.

# Saida - modelo original:
>>  I'm selfish, impatient and  =>  I'm selfish, impatient and a little bit of a control freak. I'm also a mom, a wife, a daughter,


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

'''
