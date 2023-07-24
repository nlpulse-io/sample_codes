
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


output_dir = "llama2chat_finetuned_model"
model_path = os.environ.get("model_path", output_dir)

target_model_path = "llama2chat_finetuned_model_applied"
target_model_path = os.environ.get("target_model_path", target_model_path)


# model
print("Loading the original model...")
config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, 
        use_auth_token=True)
lora_model = PeftModel.from_pretrained(base_model, model_path)

# tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path, 
        use_auth_token=True)


# https://github.com/lm-sys/FastChat/blob/833d65032a715240a3978f4a8f08e7a496c83cb1/fastchat/model/apply_lora.py#L8
print("Applying the original model...")
model = lora_model.merge_and_unload()
model.save_pretrained(target_model_path)
base_tokenizer.save_pretrained(target_model_path)
print("Gerado com sucesso:", target_model_path)

'''
# Execucao:
cd /var/server1/docker/volumes/code/project/lab1/poc_qaml2/experimentos/peft_quantization
source ~/venv/peft_quantization/bin/activate
CUDA_VISIBLE_DEVICES=1 model_path=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits target_model_path=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_applied python llama2chat_apply_finetuned_model.py
CUDA_VISIBLE_DEVICES=1 model_path=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_8gradsteps target_model_path=/var/server1/docker/volumes/shared/model/llama2chat_finetune_quantization_4bits_8gradsteps_applied python llama2chat_apply_finetuned_model.py



# Consumo:
$ nvidia-smi && free -h
Mon Jul 24 11:53:52 2023
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 525.125.06   Driver Version: 525.125.06   CUDA Version: 12.0     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   1  NVIDIA GeForce ...  Off  | 00000000:04:00.0 Off |                  N/A |
|  0%   52C    P2    33W / 170W |    500MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+
               total        used        free      shared  buff/cache   available
Mem:            77Gi        37Gi       736Mi       114Mi        39Gi        39Gi
Swap:           37Gi       4.1Gi        33Gi


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