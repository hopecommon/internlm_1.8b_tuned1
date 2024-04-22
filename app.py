import gradio as gr
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from openxlab.model import download

base_path = './try'
if not os.path.exists(base_path):
    os.system('apt install git')
    os.system('apt install git-lfs')
    os.system("git lfs install")
    os.system(f'git clone https://code.openxlab.org.cn/hopecommon/try.git {base_path}')
    os.system("git lfs install")
    os.system(f'cd {base_path} && git lfs pull')
tokenizer = AutoTokenizer.from_pretrained(base_path,trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(base_path,trust_remote_code=True, torch_dtype=torch.float16).cuda()



def chat(message,history):
    for response,history in model.stream_chat(tokenizer,message,history,max_length=2048,top_p=0.7,temperature=1):
        yield response

gr.ChatInterface(chat,
                 title="InternLM2-Chat-1.8B",
                description="""
InternLM is mainly developed by Shanghai AI Laboratory.  
                 """,
                 ).queue(1).launch()
