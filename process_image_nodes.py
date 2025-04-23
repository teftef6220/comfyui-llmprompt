import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image


def extract_pose_center(pose):
    keypoints = pose.get("pose_keypoints_2d", [])  # OpenPose形式前提
    if not keypoints or len(keypoints) < 3:
        return None
    keypoints = np.array(keypoints).reshape(-1, 3)
    valid = keypoints[:, 2] > 0.1  # confidenceでフィルタ
    if not np.any(valid):
        return None
    center = keypoints[valid, :2].mean(axis=0)
    return center

# def compute_scale(pose):
#     keypoints = np.array(pose.get("pose_keypoints_2d", [])).reshape(-1, 3)
#     valid = keypoints[:, 2] > 0.1
#     if not np.any(valid):
#         return 1.0
#     coords = keypoints[valid, :2]
#     scale = np.linalg.norm(coords.max(axis=0) - coords.min(axis=0))
#     return scale

def get_point(pose, index):
    kp = pose.get("pose_keypoints_2d", [])
    if not kp or len(kp) < 3 * (index + 1):
        return None
    x, y, conf = kp[3*index:3*index+3]
    return np.array([x, y]) if conf > 0.1 else None

def compute_scale(input_pose, ref_pose, idx1=2, idx2=5):
    """例: idx1=2(R-Shoulder), idx2=5(L-Shoulder)"""
    p1_input = get_point(input_pose, idx1)
    p2_input = get_point(input_pose, idx2)
    p1_ref   = get_point(ref_pose, idx1)
    p2_ref   = get_point(ref_pose, idx2)

    if p1_input is None or p2_input is None or p1_ref is None or p2_ref is None:
        return 1.0  # デフォルトスケール

    len_input = np.linalg.norm(p1_input - p2_input)
    len_ref   = np.linalg.norm(p1_ref - p2_ref)

    if len_ref < 1e-5:
        return 1.0
    return len_input / len_ref

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
                "threshold": ("FLOAT", {"default": 0.1, "min": 0.05, "max": 1.0}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("IMAGE",)
    FUNCTION = "fix"
    CATEGORY = "MyPromptTest"

    def fix(self, pose_images, threshold=0.1):
        T, H, W, C = pose_images.shape
        smoothed = []

        for t in range(T):
            current = pose_images[t]
            prev = pose_images[t - 1] if t > 0 else current
            next = pose_images[t + 1] if t < T - 1 else current

            # 平均ではなく max を使うことで、細い線にも反応しやすくする
            current_max = current.max(dim=-1, keepdim=True).values
            prev_max = prev.max(dim=-1, keepdim=True).values
            next_max = next.max(dim=-1, keepdim=True).values

            # current が暗い & 両隣が明るい → 消えてるとみなす
            mask = ((current_max < threshold) & (prev_max > threshold) & (next_max > threshold)).float()

            restored = (prev + next) / 2
            blended = current * (1 - mask) + restored * mask
            smoothed.append(blended.unsqueeze(0))

        result = torch.cat(smoothed, dim=0)

        # モノクロ画像なら RGB に拡張（PIL保存の互換性のため）
        if result.shape[-1] == 1:
            result = result.repeat(1, 1, 1, 3)

        return (result,)


class AlignPOSE_KEYPOINTToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_keypoints": ("POSE_KEYPOINT",),
                "input_image" : ("IMAGE", {"forceInputList": True}),
                "reference_keypoints": ("POSE_KEYPOINT", {"forceInputList": True}),
                "reference_images": ("IMAGE", {"forceInputList": True}),
                "adjust_scale": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_video",)
    # OUTPUT_IS_LIST = (True, True)
    FUNCTION = "align"
    CATEGORY = "MyPromptTest"

    def align(self, input_keypoints,input_image, reference_keypoints, reference_images, adjust_scale=True):
        input_pose = input_keypoints[0]['people'][0]
        ref_pose_0 = reference_keypoints[0]['people'][0]


        # input_image から高さ・幅を取得（1枚 only）
        if input_image[0].ndim == 3 and input_image[0].shape[0] > 4:
            input_img = input_image[0].permute(2, 0, 1)  # HWC → CHW
        else:
            input_img = input_image[0]

        _, target_H, target_W = input_img.shape

        # スケール計算（肩幅ベースなど）
        scale_factor = 1.0
        if adjust_scale:
            scale_factor = compute_scale(input_pose, ref_pose_0, idx1=2, idx2=5)

        # ref_pose をスケーリング（x, y にだけ）
        ref_pose_scaled = {
            "pose_keypoints_2d": [
                v * scale_factor if i % 3 != 2 else v  # x, y → スケーリング
                for i, v in enumerate(ref_pose_0["pose_keypoints_2d"])
            ]
        }

        input_center = extract_pose_center(input_pose)
        ref_center = extract_pose_center(ref_pose_scaled)

        if input_center is None or ref_center is None:
            raise ValueError("中心点の抽出に失敗しました")

        offset = input_center - ref_center
        dx, dy = int(round(offset[0])), int(round(offset[1]))

        aligned_images = []
        for img in reference_images:
            if img.ndim == 3 and img.shape[0] > 4:
                img = img.permute(2, 0, 1)
            elif img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            C, H, W = img.shape
            img_pil = TF.to_pil_image(img)

            # --- Scalaing ---
            if adjust_scale and scale_factor != 1.0:
                new_size = (int(W * scale_factor), int(H * scale_factor))
                img_pil = img_pil.resize(new_size, Image.BICUBIC)
                img = TF.to_tensor(img_pil)

            # --- Move in offset  ---
            C, H, W = img.shape
            shifted = torch.zeros((C, target_H, target_W), dtype=img.dtype)

            # --- 貼り付け ---
            x0 = max(0, dx)
            y0 = max(0, dy)
            x1 = min(target_W, W + dx)
            y1 = min(target_H, H + dy)

            # --- 元画像クロップ ---
            src_x0 = max(0, -dx)
            src_y0 = max(0, -dy)
            src_x1 = src_x0 + (x1 - x0)
            src_y1 = src_y0 + (y1 - y0)

            shifted[:, y0:y1, x0:x1] = img[:, src_y0:src_y1, src_x0:src_x1]
            aligned_images.append(shifted)

        video_tensor = torch.stack(aligned_images, dim=0)  # (T, C, H, W)
        aligned_video = video_tensor.permute(0, 2, 3, 1)   # → (T, H, W, C)
        return (aligned_video,)


NODE_CLASS_MAPPINGS = {
    "Rebuilt_Video": RebuiltVideo,
    "BoneImageTemporalFixer": BoneImageTemporalFixer,
    "AlignPOSE_KEYPOINTToReference": AlignPOSE_KEYPOINTToReference,
}



