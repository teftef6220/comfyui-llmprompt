import folder_paths

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
from PIL import Image
import numpy as np
import folder_paths
import subprocess
import uuid
import warnings
import shutil
from transformers import BatchEncoding




# "model_name": (folder_paths.get_filename_list("LLM"), {"tooltip": "These models are loaded from 'ComfyUI/models/LLM'"})
current_model = {}
current_qwen2VL_model_cache = {}

def recursive_to_device(obj, device):
    if isinstance(obj, torch.Tensor):
        return obj.to(device)
    elif isinstance(obj, list):
        return [recursive_to_device(x, device) for x in obj]
    elif isinstance(obj, dict):
        return {k: recursive_to_device(v, device) for k, v in obj.items()}
    else:
        return obj


def tensor_to_pil(image_tensor: torch.Tensor, frame_index: int = 0) -> Image.Image:
    """
    Converts various tensor formats to a single-frame PIL Image.
    Supports:
    - [C, H, W]
    - [H, W, C]
    - [1, C, H, W]
    - [1, H, W, C]
    - [T, C, H, W] (video)
    - [T, H, W, C] (video)
    """

    # Remove batch dim if present
    image_tensor = image_tensor.squeeze()

    if image_tensor.ndim == 4:
        # video: TCHW or THWC
        if image_tensor.shape[1] in [1, 3]:  # TCHW
            image_tensor = image_tensor[frame_index]  # [C, H, W]
            image_tensor = image_tensor.permute(1, 2, 0)  # → HWC
        else:  # THWC
            image_tensor = image_tensor[frame_index]  # [H, W, C]

    elif image_tensor.ndim == 3:
        if image_tensor.shape[0] in [1, 3]:  # CHW
            image_tensor = image_tensor.permute(1, 2, 0)  # → HWC
        elif image_tensor.shape[2] in [1, 3, 4]:  # HWC
            pass  # OK
        else:
            raise ValueError(f"Ambiguous 3D tensor shape: {image_tensor.shape}")

    else:
        raise ValueError(f"Unsupported tensor shape: {image_tensor.shape}")

    image_np = image_tensor.detach().cpu().numpy()
    if image_np.max() <= 1.0:
        image_np = image_np * 255.0

    image_np = np.clip(image_np, 0, 255).astype(np.uint8)

    return Image.fromarray(image_np)

class Loadllm:
    @classmethod
    def INPUT_TYPES(cls):
        llm_base_path_list = folder_paths.get_folder_paths("LLM")
        llm_base_path = llm_base_path_list[0] if isinstance(llm_base_path_list, list) else llm_base_path_list
        
        try:
            choices = [
                name for name in os.listdir(llm_base_path)
                if os.path.isdir(os.path.join(llm_base_path, name))
                and any(f.endswith(".safetensors") for f in os.listdir(os.path.join(llm_base_path, name)))
            ]
        except Exception as e:
            choices = []

        return {
            "required": {
                "model_name": (tuple(choices),),
                "precision": (["fp32", "bf16", "fp16"],
                    {"default": "fp16"}
                ),
            },
            "optional": {
                "device": (["cuda", "cpu"], {"default": "cuda"}),
                "quantization": (['disabled', 'fp8_e4m3fn'], {"default": 'disabled', "tooltip": "optional quantization method"}),
            }
        }
    
    RETURN_TYPES = ("LLM_Model",)
    RETURN_NAMES = ("model",)
    FUNCTION = "load"
    CATEGORY = "custom/llm_nodes"

    def load(self, model_name, precision, device="cuda", quantization="disabled"):
        global current_model

        # model_path = folder_paths.get_full_path("LLM", model_name)
        model_path = os.path.join(folder_paths.get_folder_paths("LLM")[0],model_name)

        key = (model_path, precision, device)

        if key in current_model:
            print(f"[INFO] Reusing cached model: {model_name}")
            return (current_model[key],)

        print(f"[INFO] Loading model: {model_path} on {device} with {precision}")

        dtype_map = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype_map[precision],
            device_map=device,
            trust_remote_code=True,
            local_files_only=True
        )
        model.eval()

        result = {
            "model_name": model_name,
            "model" : model,
            "precision" : precision,
            "tokenizer": tokenizer,
            "device": device,
        }

        # キャッシュに保存
        current_model[key] = result

        return (result,)


class GenerateTextWithLLM:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "llm_model": ("LLM_Model",),
                "prompt": ("STRING", {"multiline": True}),
                "max_tokens": ("INT", {"default": 50}),
                "temperature": ("FLOAT", {"default": 1.0}),
            }
        }
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "generate"
    CATEGORY = "custom/llm_nodes"

    def generate(self, llm_model, prompt, max_tokens=50, temperature=1.0):

        tokenizer = llm_model["tokenizer"]
        model = llm_model["model"]
        device = llm_model["device"]

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        return (generated_text,)


class Qwen2VL:

    def __init__(self):
        torch.set_default_device("cuda:0")
        self.model_checkpoint = None
        self.processor = None
        self.model = None
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        self.bf16_support = (
            torch.cuda.is_available()
            and torch.cuda.get_device_capability(self.device)[0] >= 8
        )

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"default": "", "multiline": True}),
                "model": (
                    [
                        "Qwen2_5-VL-3B-Instruct",
                        "Qwen2_5-VL-7B-Instruct",
                    ],
                    {"default": "Qwen2_5-VL-7B-Instruct"},
                ),
                "quantization": (
                    ["none", "4bit", "8bit"],
                    {"default": "none"},
                ),
                "reload_model": ("BOOLEAN", {"default": False}),
                "del_model_from_GPU": ("BOOLEAN", {"default": False}),
                "temperature": (
                    "FLOAT",
                    {"default": 0.7, "min": 0, "max": 1, "step": 0.1},
                ),
                "max_new_tokens": (
                    "INT",
                    {"default": 512, "min": 128, "max": 2048, "step": 1},
                ),
                "seed": ("INT", {"default": -1 ,"min": -1}),
            },
            "optional": {
                "max_video_frame": (
                    "INT",
                    {"default": 32, "min": 1, "max": 2048, "step": 1},
                ),
                "image": ("IMAGE",), ## torch.tensor
                # "video_path": ("STRING", {"default": ""}),
                "video_path": ("IMAGE", {"default": ""}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("generated_text",)
    FUNCTION = "inference"
    CATEGORY = "custom/llm_nodes"

    def inference(
        self,
        text,
        model,
        quantization,
        reload_model,
        del_model_from_GPU,
        temperature,
        max_new_tokens,
        seed,
        max_video_frame,
        image=None,
        video_path=None,
    ):
        global current_qwen2VL_model_cache
        if seed != -1:
            torch.manual_seed(seed)
        model_id = f"qwen/{model}"
        # put downloaded model to model/LLM dir

        for i in range(len(folder_paths.get_folder_paths("LLM"))):
            if os.path.exists(folder_paths.get_folder_paths("LLM")[i]):
                self.model_checkpoint = os.path.join(folder_paths.get_folder_paths("LLM")[i],model)
                break


        if not os.path.exists(self.model_checkpoint):
            warnings.warn("No model checkpoint found.")
            print("No Model")

        # リロードするかどうかの確認モデルと quantixe が変わったらリロード
        model_key = (model, quantization)
        if (model_key in current_qwen2VL_model_cache) :
            print(f"[INFO] Reusing model:============================= {model_key}")
            model_instance, processor = current_qwen2VL_model_cache[model_key]
            self.model = model_instance                 
            self.processor = processor  

        if (reload_model == True) or (model_key not in current_qwen2VL_model_cache) :
            print("-------------------------------Reload!!!!!!!!!!!!!!!!")

        # if self.processor is None:
            # Define min_pixels and max_pixels:
            # Images will be resized to maintain their aspect ratio
            # within the range of min_pixels and max_pixels.
            min_pixels = 256*28*28
            max_pixels = 1024*28*28 

            self.processor = AutoProcessor.from_pretrained(
                self.model_checkpoint,
                min_pixels=min_pixels,
                max_pixels=max_pixels,
                local_files_only=True
            )

        # if self.model is None:
            # Load the model on the available device(s)
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                )
            else:
                quantization_config = None

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_checkpoint,
                torch_dtype=torch.bfloat16 if self.bf16_support else torch.float16,
                device_map="auto",
                quantization_config=quantization_config,
                local_files_only=True,
                attn_implementation="eager",
            )

            # キャッシュ
            current_qwen2VL_model_cache[model_key] = (self.model, self.processor)

        with torch.no_grad():
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text},
                    ],
                }
            ]

            if (image is not None) and (video_path is not None):
                warnings.warn("Chose Image or Video only 1.")
                print("Chose Image or Video only 1")
                raise ValueError("Chose Image or Video only 1.")


            else:

                if image is not None:
                    print("deal image")
                    pil_image = tensor_to_pil(image)
                    messages[0]["content"].insert(0, {
                        "type": "image",
                        "image": pil_image,
                    })

                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    print("deal messages", messages)
                    image_inputs, video_inputs = process_vision_info(messages)## 試験的にreturn_video_kwargs=True入れてる

                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        fps = 16.0,
                        padding=True,
                        return_tensors="pt",
                    ).to("cuda")
                
                if video_path is not None: ##### Load Video Upload に対応、TODO Load Video Uploadは動画を Imagelist に分離しているので、わざわざ分離しなくても行けるように新しく Loadvideo ノード作るかも 

                    print("[INFO] Writing image sequence to temporary video")
                    unique_id = uuid.uuid4().hex
                    tmp_dir = f"./custom_nodes/comfyui-llmprompt/tmp/qwen_frames_{unique_id}"
                    os.makedirs(tmp_dir, exist_ok=True)

                    ##試験的に追加、最初の n フレームだけ取り出す

                    image_frames = video_path[:max_video_frame] 

                    # フレーム 書き出し
                    for idx, frame_tensor in enumerate(image_frames):
                        frame_image = tensor_to_pil(frame_tensor)
                        frame_image.save(os.path.join(tmp_dir, f"frame_{idx:04d}.png"))



                    # mp4 を生成
                    processed_video_path = f"./custom_nodes/comfyui-llmprompt/tmp/processed_video_{unique_id}.mp4"
                    print(processed_video_path,"DDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")

                    ffmpeg_command = [
                        "ffmpeg",
                        "-framerate", "16",  # fps=1に相当
                        "-i", os.path.join(tmp_dir, "frame_%04d.png"),
                        "-vf", "scale='min(256,iw)':min'(256,ih)':force_original_aspect_ratio=decrease",
                        "-c:v", "libx264",
                        "-preset", "fast",
                        "-crf", "18",
                        processed_video_path
                    ]
                    subprocess.run(ffmpeg_command, check=True)

                    # shutil.rmtree(tmp_dir)



                    # 添加处理后的视频信息到消息
                    messages[0]["content"].insert(0, {
                        "type": "video",
                        "video": processed_video_path,
                    })

            # 准备输入

                    text = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    print("deal messages", messages)
                    image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)## 試験的にreturn_video_kwargs=True入れてる
                    video_inputs = [v.to("cuda") for v in video_inputs]
                    inputs = self.processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                        **video_kwargs,
                    ).to("cuda")


            #
            # inputs = BatchEncoding(recursive_to_device(dict(inputs_raw), self.device))

            # 推論 # TODO GPU に乗らない！！！！！！

            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            result = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
                temperature=temperature,
            )
            # except Exception as e:
            #     return (f"Error during model inference: {str(e)}",)

           
            # 删除临时视频文件
            if video_path is not None and len(video_path) > 0:
                os.remove(processed_video_path)
                shutil.rmtree(tmp_dir)


            if del_model_from_GPU: ## Remove cache from GPU
                print("===========del model From GPU Memory =============")
                # if quantization == "8bit" :
                #     pass
                # elif  quantization == "4bit" :
                #     pass
                # else:
                #     self.model.to("cpu")
                del self.model
                del self.processor
                torch.cuda.empty_cache()

                if model_key in current_qwen2VL_model_cache:
                    del current_qwen2VL_model_cache[model_key]

            return result


NODE_CLASS_MAPPINGS = {
    "Load llm": Loadllm,
    "Generate Text with LLM": GenerateTextWithLLM,
    "Inference Qwen2VL":Qwen2VL,
}


# class Processinllm:
