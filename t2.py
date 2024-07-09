from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
from PIL import Image

def clear_cuda_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

clear_cuda_cache()
# Load model and tokenizer
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Process image function
def process_image(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'draw and detect the damages in a box  on floor or in walls  in the picture?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    image_with_bbox = tokenizer.draw_bbox_on_latest_picture(response, history)
    
    # Save the image with bounding boxes
    output_image_path = os.path.join("output", os.path.basename(image_path))
    image_with_bbox.save(output_image_path)
    
    print(f"Output image path: {output_image_path}")  # Debugging line
    print(f"text{response}")
    return response, output_image_path


def process_image_t(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'describe the damages detected appears in the image'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
  
    return response







def process_image_2(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'draw a box on damage type Scratch in picture?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    image_with_bbox = tokenizer.draw_bbox_on_latest_picture(response, history)
    
    # Save the image with bounding boxes
    output_image_path = os.path.join("output", os.path.basename(image_path))
    image_with_bbox.save(output_image_path)
    
    print(f"Output image path: {output_image_path}")  # Debugging line
    print(f"text{response}")
    return response, output_image_path



def process_image_3(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'draw a box on damage type stains in picture?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    image_with_bbox = tokenizer.draw_bbox_on_latest_picture(response, history)
    
    # Save the image with bounding boxes
    output_image_path = os.path.join("output", os.path.basename(image_path))
    image_with_bbox.save(output_image_path)
    
    print(f"Output image path: {output_image_path}")  # Debugging line
    print(f"text{response}")
    return response, output_image_path
