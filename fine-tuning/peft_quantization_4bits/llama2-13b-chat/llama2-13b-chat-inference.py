import torch
from transformers import BitsAndBytesConfig, pipeline, LlamaTokenizer, LlamaForCausalLM
from huggingface_hub import login

token = "hf_..." # altere o seu token ou leia de uma variável de ambiente
login(token=token)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

model_id = "meta-llama/Llama-2-13b-chat-hf"

model_4bit = LlamaForCausalLM.from_pretrained(
        model_id,
        # device_map="auto",
        device_map={"":0},
        quantization_config=quantization_config, use_auth_token=True)

tokenizer = LlamaTokenizer.from_pretrained(model_id, use_auth_token=True)

pipeline = pipeline("text-generation", model=model_4bit, tokenizer=tokenizer, use_auth_token=True)

prompt = """
Instrução:
Considerando a tabela com o resultado por mes por produto, me apresente alguns insights sobre o resultado da minha loja.
Sinalize quais foram os melhores resultados e os principais pontos de atenção.

Tabela de vendas por produto:
mes,ano,produto,unidades_vendidas,ticket_medio,total_venda,total_custo,resultado_operacional
1,2023,tenis,74,322.8706757,23892.43,19113.944,4778.486
2,2023,tenis,115,282.2514783,32458.92,16229.46,16229.46
3,2023,tenis,82,315.535,25873.87,23286.483,2587.387
4,2023,tenis,74,322.6051351,23872.78,14323.668,9549.112
5,2023,tenis,11,352.02,3872.22,2710.554,1161.666
6,2023,tenis,94,309.3424468,29078.19,17446.914,11631.276
1,2023,sapatenis,37,350.8681081,12982.12,7789.272,5192.848
2,2023,sapatenis,36,330.2861111,11890.3,7728.695,4161.605
3,2023,sapatenis,30,327.404,9822.12,6875.484,2946.636
4,2023,sapatenis,35,360.4805714,12616.82,6813.0828,5803.7372
5,2023,sapatenis,34,349.18,11872.12,7360.7144,4511.4056
6,2023,sapatenis,30,362.4113333,10872.34,5653.6168,5218.7232
1,2023,sapato,97,224.2703093,21754.22,7831.5192,13922.7008
2,2023,sapato,113,246.8622124,27895.43,11995.0349,15900.3951
3,2023,sapato,100,234.899,23489.9,8926.162,14563.738
4,2023,sapato,115,230.623913,26521.75,10343.4825,16178.2675
5,2023,sapato,107,232.6371028,24892.17,9459.0246,15433.1454
6,2023,sapato,103,246.4215534,25381.42,10152.568,15228.852
"""

max_new_tokens = 720
sequences = pipeline(prompt,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_k=10,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
        )

print("Result_3:");
for sequence in sequences:
    print(sequence['generated_text'])
