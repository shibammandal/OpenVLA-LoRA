import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

model_id = "openvla/openvla-7b"

processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)

model = AutoModelForImageTextToText.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True,
)

print(model)
print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}" )
