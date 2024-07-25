from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import torch
import os
from PIL import Image
from cali import calib_pixel
import re



# Load model and tokenizer
torch.manual_seed(1234)
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="auto", trust_remote_code=True, bf16=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)


def process_image_t(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'describe the damages detected appears in the image'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)

    return response

def process_image_new(image_path):
    query = tokenizer.from_list_format([
        {'image': image_path},  # Local path
        {'text': 'draw and detect the damages in a box in the picture?'},
    ])
    response, history = model.chat(tokenizer, query=query, history=None)
    image_with_bbox = tokenizer.draw_bbox_on_latest_picture(response, history)

    output_image_path = os.path.join("output", os.path.basename(image_path))
    image_with_bbox.save(output_image_path)


    # Save the response text in a .txt file with the same name as the input image
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_text_path = os.path.join("output", f"{base_name}.txt")

    print(f"Output image path: {output_image_path}")  # Debugging line
    print(f"Output text path: {output_text_path}")    # Debugging line
    print(f"text: {response}")

    # Extract bounding box dimensions
    width, height = extract_bbox_dimensions(response)
    print(f"Bounding box width: {width}, height: {height}")

   # square_pixel_distance = calib_pixel()
    square_pixel_distance = 123
    real_width = compute_real_distance(width,square_pixel_distance)
    real_length = compute_real_distance(height,square_pixel_distance)
    
    with open(output_text_path, 'w') as file:
        file.write(f"Width: {real_width}\n")
        file.write(f"Height: {real_length}\n")

    return response, output_image_path,real_width,real_length

def extract_bbox_dimensions(response):
    # Regex to find the bounding box coordinates
    bbox_pattern = re.compile(r'<box>\((\d+),(\d+)\),\((\d+),(\d+)\)</box>')
    match = bbox_pattern.search(response)

    if match:
        x1, y1, x2, y2 = map(int, match.groups())
        width = x2 - x1
        height = y2 - y1
        return width, height
    
def compute_real_distance(pixel_distance, square_pixel_distance):
    """
    Compute the real-world distance based on pixel distance.

    Parameters:
    pixel_distance (): Distance between two points in pixels.
    square_length (): Real-world length of one side of the checkerboard square (in mm).
    square_pixel_distance (): Pixel distance representing the length of one square side in pixels.

    Returns:
    float: Real-world distance corresponding to the pixel distance.
    """
    square_length = 22

    # Compute the real-world distance
    real_distance = (pixel_distance * square_length) / square_pixel_distance
    
    return real_distance

