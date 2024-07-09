import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval().to(device)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Configure generation settings
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# 1st dialogue turn
query = tokenizer.from_list_format([
    {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},  # Either a local path or an URL
    {'text': 'what is it?'},
])

response, history = model.chat(tokenizer, query=query, history=None)
print(response)
