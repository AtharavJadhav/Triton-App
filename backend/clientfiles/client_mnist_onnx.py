import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import sys
import os

# Define the server URL and model name
url = "localhost:8000"
model_name = "mnist_model_onnx"

# Create a Triton HTTP client
client = httpclient.InferenceServerClient(url=url)

# Prepare a sample image (MNIST digit)
def prepare_image(image_path):
    img = Image.open(image_path).convert('L')
    img = img.resize((28, 28), Image.LANCZOS)
    img_np = np.array(img).astype(np.float32)
    img_np = (img_np / 255.0).reshape(1, 1, 28, 28)
    return img_np

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 client_mnist_onnx.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]

    if not os.path.exists(image_path):
        print(f"Error: File {image_path} does not exist.")
        sys.exit(1)

    # Prepare the image
    image = prepare_image(image_path)

    # Create the input object
    inputs = httpclient.InferInput("input.1", image.shape, "FP32")
    inputs.set_data_from_numpy(image)

    # Create the output object
    outputs = httpclient.InferRequestedOutput("18")

    # Send the request to the server
    response = client.infer(model_name, inputs=[inputs], outputs=[outputs])

    # Get the results
    result = response.as_numpy("18")
    predicted_digit = np.argmax(result)

    # Write the result to results.txt
    results_dir = "inference_results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    results_file_path = os.path.join(results_dir, "results.txt")
    with open(results_file_path, "w") as f:
        f.write(f"Predicted Digit: {predicted_digit}\n")
