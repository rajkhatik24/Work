import gradio as gr
import time
from transformers import pipeline
from PIL import Image
import requests
from io import BytesIO

# Define the depth estimation models you want to compare
models = {
    "Depth Anything": "LiheYoung/depth-anything-small-hf",
    "Intel Dpt": "Intel/dpt-large",
    # Add more models as needed
}

# Function to perform depth estimation
def estimate_depth(model_name, img):
    # Load the selected model from the pipeline
    depth_estimator = pipeline("depth-estimation", model=models[model_name], device= "cuda")
    
    # Start timing
    start_time = time.time()
    
    # Perform depth estimation
    result = depth_estimator(img)
    
    # End timing
    end_time = time.time()
    
    # Calculate the time taken
    time_taken = end_time - start_time
    
    # Extract the depth map
    depth_map = result['depth']
    time_taken =f"Time taken: {time_taken:.2f} seconds"
    # Return the depth map and the time taken
    return depth_map, time_taken

# Example images for testing
example_images = [
    [ "https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/dog-puppy-on-garden-royalty-free-image-1586966191.jpg"],
    ["https://images.unsplash.com/photo-1704347959552-93f803ea10f7?q=80&w=2070&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D.jpg"],
    ["https://plus.unsplash.com/premium_photo-1664298849700-abbfe6f854d5?q=80&w=1887&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D.jpg"],
    # Add more example model-image pairs if needed
]

# Download example images and convert them to PIL images
def download_image(url):
    response = requests.get(url)
    return Image.open(BytesIO(response.content))


with gr.Blocks() as demo:
    with gr.Row():
        input_image = gr.Image(type="pil", label="Input Image")
        processed_frames = gr.Image(type="pil", label="Depth Map")
        

    with gr.Row():
        drop = gr.Dropdown(list(models.keys()), label="Select Model", allow_custom_value=True)
        response = gr.Text(label="Time Taken")
    with gr.Row():
        process_video_btn = gr.Button("process image")
    with gr.Row():    
        examples = gr.Examples(example_images, inputs=input_image)

    process_video_btn.click(estimate_depth, [drop,input_image], [processed_frames, response])

demo.queue()
demo.launch()
# Launch the interface
