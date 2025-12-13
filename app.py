import sys
import types

# Fix Python 3.12 UnionType incompatibility with transformers
if sys.version_info >= (3, 12):
    # Patch transformers before importing model classes
    from transformers.utils import auto_docstring
    
    # Get the actual function from the module
    original_process_param_type = auto_docstring._process_parameter_type
    
    def patched_process_parameter_type(param, param_name, func):
        """Fixed version that handles Python 3.12 UnionType"""
        try:
            return original_process_param_type(param, param_name, func)
        except AttributeError as e:
            if "UnionType" in str(e) and "__name__" in str(e):
                # Handle UnionType which doesn't have __name__
                if isinstance(param.annotation, types.UnionType):
                    # Extract union args and format them
                    args = param.annotation.__args__
                    arg_names = []
                    for arg in args:
                        if hasattr(arg, '__name__'):
                            arg_names.append(arg.__name__)
                        elif hasattr(arg, '__module__'):
                            arg_names.append(f"{arg.__module__}.{str(arg).split('.')[-1]}")
                        else:
                            arg_names.append(str(arg))
                    
                    param_type = f"Union[{', '.join(arg_names)}]"
                    optional = type(None) in args
                    return param_type, optional
            raise
    
    # Apply the patch
    auto_docstring._process_parameter_type = patched_process_parameter_type
    print("✓ Applied Python 3.12 compatibility patch for transformers")

from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
print(f"Loading model: {model_name}")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

print("✓ Model loaded successfully!")

messages = [
    {"role": "system", "content": "You are a helpful assistant provided by Moonshot-AI."},
    {"role": "user", "content": "Is 123 a prime?"}
]
input_ids = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True, 
    return_tensors="pt"
).to(model.device)

print("Generating response...")
generated_ids = model.generate(inputs=input_ids, max_new_tokens=500)
response = tokenizer.batch_decode(generated_ids)[0]
print("\n" + "="*50)
print(response)
print("="*50)
