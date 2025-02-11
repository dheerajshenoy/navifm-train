from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import load_file

# Define model path
model_path = "./tinyllama"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Load model using SafeTensors
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    trust_remote_code=True,
    use_safetensors=True
)

# Test the model
inputs = tokenizer("Hello, AI!", return_tensors="pt")
outputs = model.generate(**inputs)

if (outputs != None):
    print(tokenizer.decode(outputs[0]))
