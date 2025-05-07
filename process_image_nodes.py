import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image


def extract_pose_center(pose):
    keypoints = pose.get("pose_keypoints_2d", [])
    if not keypoints or len(keypoints) < 3:
        return None
    keypoints = np.array(keypoints).reshape(-1, 3)
    neck = keypoints[1]
    return neck[:2] if neck[2] > 0.1 else None


def get_point(pose, index):
    kp = pose.get("pose_keypoints_2d", [])
    if not kp or len(kp) < 3 * (index + 1):
        return None
    x, y, conf = kp[3 * index:3 * index + 3]
    return np.array([x, y]) if conf > 0.1 else None


def compute_scale(input_pose, ref_pose, idx1=2, idx2=5):
    """例: idx1=2(R-Shoulder), idx2=5(L-Shoulder)

    型の欠損に頑健な設計に改造した
    片方の肩が欠損している場合
    - 右肩が欠損している場合 : 左肩のスケールのみを使用して結果を返す
    - 左肩が欠損している場合 : 右肩のスケールのみを使用して結果を返す
    - 両肩が欠損している場合 : デフォルト値 `1.0` を返す
    ]"""
    neck_idx = 1

    def length(p1, p2):
        return np.linalg.norm(p1 - p2)

    def safe_scale(p_neck_input, p_shoulder_input, p_neck_ref, p_shoulder_ref):
        if any(p is None for p in [p_neck_input, p_shoulder_input, p_neck_ref, p_shoulder_ref]):
            return None
        len_input = length(p_neck_input, p_shoulder_input)
        len_ref = length(p_neck_ref, p_shoulder_ref)
        return len_input / len_ref if len_ref > 1e-5 else None

    p_neck_input = get_point(input_pose, neck_idx)
    p_neck_ref = get_point(ref_pose, neck_idx)

    scale_r = safe_scale(p_neck_input, get_point(input_pose, idx1), p_neck_ref, get_point(ref_pose, idx1))
    scale_l = safe_scale(p_neck_input, get_point(input_pose, idx2), p_neck_ref, get_point(ref_pose, idx2))

    scales = [s for s in [scale_r, scale_l] if s is not None]
    if not scales:
        raise ValueError("肩または首のキーポイントが欠損しているか、距離が小さすぎます")
    return sum(scales) / len(scales)


def scale_with_anchor(img_tensor, pose_keypoints, scale_factor, ref_anchor_target, anchor_index=1):
    C, H, W = img_tensor.shape

    keypoints = np.array(pose_keypoints).reshape(-1, 3)
    anchor = keypoints[anchor_index][:2]
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    new_size = (new_H, new_W)

    # torchでリサイズ
    img_tensor_unsq = img_tensor.unsqueeze(0)  # [1, C, H, W]
    img_scaled_tensor = F.interpolate(img_tensor_unsq, size=new_size, mode="bicubic", align_corners=False)
    img_scaled_tensor = img_scaled_tensor[0]  # [C, H, W]

    new_anchor = anchor * scale_factor
    shift = (ref_anchor_target - new_anchor).astype(int)

    canvas = torch.zeros((C, H, W), dtype=img_tensor.dtype)

    paste_x0 = max(0, shift[0])
    paste_y0 = max(0, shift[1])
    crop_x0 = max(0, -shift[0])
    crop_y0 = max(0, -shift[1])

    paste_h = min(H - paste_y0, img_scaled_tensor.shape[1] - crop_y0)
    paste_w = min(W - paste_x0, img_scaled_tensor.shape[2] - crop_x0)

    canvas[:, paste_y0:paste_y0+paste_h, paste_x0:paste_x0+paste_w] = \
        img_scaled_tensor[:, crop_y0:crop_y0+paste_h, crop_x0:crop_x0+paste_w]
    return canvas

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
                "input_image": ("IMAGE", {"forceInputList": True}),
                "input_keypoints": ("POSE_KEYPOINT",),
                "reference_images": ("IMAGE", {"forceInputList": True}),
                "reference_keypoints": ("POSE_KEYPOINT", {"forceInputList": True}),
                "adjust_scale": ("BOOLEAN", {"default": True}),
                "custom_resize_scale": ("BOOLEAN", {"default": False}),
                "custom_scaling_factor": ("FLOAT",{"default": 0.1, "min": 0.05, "max": 100.0})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_video",)
    FUNCTION = "align"
    CATEGORY = "MyPromptTest"

    def align(self, input_keypoints, input_image, reference_keypoints, reference_images, adjust_scale=True, custom_resize_scale = False, custom_scaling_factor = 1.0):
        input_pose = input_keypoints[0]['people'][0]
        ref_pose_0 = reference_keypoints[0]['people'][0]

        input_img = input_image[0].permute(2, 0, 1) if input_image[0].ndim == 3 else input_image[0]
        _, target_H, target_W = input_img.shape

        input_center = extract_pose_center(input_pose)
        ref_center = extract_pose_center(ref_pose_0)
        if input_center is None or ref_center is None:
            raise ValueError("中心点の抽出に失敗しました")

        scale_factor = compute_scale(input_pose, ref_pose_0) if adjust_scale else 1.0

        if custom_resize_scale:
            scale_factor  = scale_factor  * custom_scaling_factor

        aligned_images = []
        for ref_img in reference_images:
            img_tensor = ref_img.permute(2, 0, 1) if ref_img.ndim == 3 else ref_img
            aligned = scale_with_anchor(img_tensor, ref_pose_0["pose_keypoints_2d"], scale_factor, input_center)
            aligned_images.append(aligned)

        video_tensor = torch.stack(aligned_images, dim=0).permute(0, 2, 3, 1)
        return (video_tensor,)




NODE_CLASS_MAPPINGS = {
    "Rebuilt_Video": RebuiltVideo,
    "BoneImageTemporalFixer": BoneImageTemporalFixer,
    "AlignPOSE_KEYPOINTToReference": AlignPOSE_KEYPOINTToReference,
}



