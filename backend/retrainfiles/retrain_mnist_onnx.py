import numpy as np
from PIL import Image
import sys
import os
import mlflow
import mlflow.pytorch
import mlflow.onnx
import onnx #type: ignore
import onnxruntime #type: ignore
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import shutil

# Define the neural network model
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = torch.max_pool2d(self.conv2(x), 2)
        x = torch.relu(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        return torch.log_softmax(x, dim=1)

# Training settings
batch_size = 64
epochs = 5
lr = 0.01
momentum = 0.5
log_interval = 10

# Initialize data loaders
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
train_loader = DataLoader(datasets.MNIST('../data', train=True, download=True, transform=transform), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(datasets.MNIST('../data', train=False, transform=transform), batch_size=batch_size, shuffle=False)

# Initialize model, loss function, and optimizer
model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

def train_model():
    mlflow.set_experiment("MNIST_ONNX_Experiment")
    with mlflow.start_run() as run:
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("momentum", momentum)

        for epoch in range(epochs):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if batch_idx % log_interval == 0:
                    mlflow.log_metric('loss', loss.item(), step=epoch * len(train_loader) + batch_idx)

            # Evaluate on test set
            model.eval()
            test_loss = 0
            correct = 0
            with torch.no_grad():
                for data, target in test_loader:
                    output = model(data)
                    test_loss += criterion(output, target).item()
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()

            test_loss /= len(test_loader.dataset)
            accuracy = 100. * correct / len(test_loader.dataset)
            mlflow.log_metric('test_loss', test_loss, step=epoch)
            mlflow.log_metric('accuracy', accuracy, step=epoch)

    print('Training complete.')
    return model

trained_model = train_model()

model_scripted = torch.jit.script(trained_model) # Export to TorchScript

model_scripted.save('model.pt') # Save

def load_pytorch_model(filepath):
    model = torch.load(filepath)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to convert PyTorch model to ONNX
def convert_pytorch_to_onnx(model, onnx_filepath, input_shape):
    dummy_input = torch.randn(*input_shape)
    torch.onnx.export(model, dummy_input, onnx_filepath, opset_version=11)

# Function to enable dynamic batching
def enable_dynamic_batching(onnx_filepath):
    model = onnx.load(onnx_filepath)
    graph = model.graph
    for input_tensor in graph.input:
        input_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    for output_tensor in graph.output:
        output_tensor.type.tensor_type.shape.dim[0].dim_param = 'batch_size'
    onnx.save(model, onnx_filepath)

# Function to get model input and output names
def get_model_io_names(onnx_filepath):
    model = onnx.load(onnx_filepath)
    input_name = model.graph.input[0].name
    output_name = model.graph.output[0].name
    input_shape = [dim.dim_value for dim in model.graph.input[0].type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in model.graph.output[0].type.tensor_type.shape.dim]
    return input_name, output_name, input_shape, output_shape

# Function to prepare Triton repository
def prepare_triton_repository(onnx_filepath, model_name):
    model_dir = f"model_repository/{model_name}"
    os.makedirs(model_dir, exist_ok=True)
    
    # Find the next available directory number
    existing_dirs = [int(d) for d in os.listdir(model_dir) if d.isdigit()]
    next_dir_number = max(existing_dirs, default=0) + 1
    
    model_version_dir = os.path.join(model_dir, str(next_dir_number))
    os.makedirs(model_version_dir)
    
    shutil.move(onnx_filepath, os.path.join(model_version_dir, "model.onnx"))

    input_name, output_name, input_shape, output_shape = get_model_io_names(os.path.join(model_version_dir, "model.onnx"))

    # Modify input_shape and output_shape to have -1 as the first dimension
    input_shape[0] = -1
    output_shape[0] = -1

    with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
        f.write(f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0  # Enable dynamic batching
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: {input_shape}  # Excluding the dynamic batch dimension
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: {output_shape}  # Excluding the dynamic batch dimension
  }}
]
        """)

model_path = "/home/atharav/react/model/Triton-App/backend/model.pt"  # Specify your model file path
model_name = "mnist_model_onnx"
input_shape = (1, 1, 28, 28)  # Update with actual input shape for your model

model = load_pytorch_model(model_path)
onnx_filepath = "model.onnx"
convert_pytorch_to_onnx(model, onnx_filepath, input_shape)

enable_dynamic_batching(onnx_filepath)
prepare_triton_repository(onnx_filepath, model_name)
print("Model conversion, dynamic batching, and Triton repository preparation complete.")

os.remove("model.pt")