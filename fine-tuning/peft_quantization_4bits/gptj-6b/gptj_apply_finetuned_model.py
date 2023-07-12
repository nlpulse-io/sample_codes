
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftConfig, PeftModel


output_dir = "gptj_finetuned_model"
model_path = os.environ.get("model_path", output_dir)

target_model_path = "gptj_finetuned_model_applied"
target_model_path = os.environ.get("target_model_path", target_model_path)


# model
print("Loading the original model...")
config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
lora_model = PeftModel.from_pretrained(base_model, model_path)

# tokenizer
base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)


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
CUDA_VISIBLE_DEVICES=1 model_path=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits target_model_path=/var/server1/docker/volumes/shared/model/gptj_finetune_quantization_4bits_applied python gptj_apply_finetuned_model.py


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
|  0%   48C    P8    14W / 170W |    495MiB / 12288MiB |      0%      Default |
|                               |                      |                  N/A |
+-------------------------------+----------------------+----------------------+

'''