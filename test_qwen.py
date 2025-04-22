

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
    BitsAndBytesConfig,
)
from qwen_vl_utils import process_vision_info

torch.set_default_device("cuda:0")
model_path = "/mnt/blue1/cho/SD_Models/models/LLM/Qwen2_5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="float16", device_map="auto",local_files_only=True
)

processor = AutoProcessor.from_pretrained(
    model_path,
    local_files_only=True
)
# Messages containing a local video path and a text query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/home/cho/Code/ComfyUI/custom_nodes/comfyui-llmprompt/tmp/processed_video_f3890baf1e5840caad749734540fa8c7.mp4",
                "max_pixels": 360 * 420,
                "fps": 16.0,
            },
            {"type": "text", "text": "Describe this video."},
        ],
    }
]


#In Qwen 2.5 VL, frame rate information is also input into the model to align with absolute time.
# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    # fps="1.0",
    padding=True,
    return_tensors="pt",
    **video_kwargs,
)
inputs = inputs.to("cuda")

# Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)