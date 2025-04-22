import torch
import torch.nn.functional as F
import numpy as np



class RebuiltVideo:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "repeat_each_frame": ("INT", {"default": 3, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "width", "height",)
    FUNCTION = "rebuilt"
    CATEGORY = "MyPromptTest"

    def rebuilt(self, image, repeat_each_frame):
        # image: [B, H, W, C] (B = number of frames)
        B, H, W, C = image.shape

        # 複製用テンソル作成: repeat along batch/frame dimension
        image_expanded = image.repeat_interleave(repeat_each_frame, dim=0)

        return image_expanded, W, H





class BoneImageTemporalFixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "pose_images": ("IMAGE",),  # [T, H, W, C]
                "blur_window": ("INT", {"default": 3, "min": 1, "max": 11}),
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "fix"
    CATEGORY = "MyPromptTest"

    def fix(self, pose_images, blur_window, threshold):
        T, H, W, C = pose_images.shape
        pad = blur_window // 2

        start_pad = pose_images[0:1].repeat(pad, 1, 1, 1)
        end_pad = pose_images[-1:].repeat(pad, 1, 1, 1)
        padded = torch.cat([start_pad, pose_images, end_pad], dim=0)

        smoothed = []
        for t in range(T):
            current = pose_images[t]
            window = padded[t:t + blur_window]
            mean_img = window.mean(dim=0)
            diff = torch.abs(current - mean_img).mean(dim=-1, keepdim=True)
            mask = (diff > threshold).float()
            blended = current * (1 - mask) + mean_img * mask
            smoothed.append(blended.unsqueeze(0))

        result = torch.cat(smoothed, dim=0)  # [T, H, W, C]

        if result.shape[-1] == 1:
            result = result.repeat(1, 1, 1, 3)  # [T, H, W, 1] → [T, H, W, 3]

        return (result,)




NODE_CLASS_MAPPINGS = {
    "Rebuilt_Video": RebuiltVideo,
    "BoneImageTemporalFixer": BoneImageTemporalFixer,
}



