import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import numpy as np
import cv2
from PIL import Image


# def extract_pose_center(pose):
#     keypoints = pose.get("pose_keypoints_2d", [])  # OpenPose形式前提
#     if not keypoints or len(keypoints) < 3:
#         return None
#     keypoints = np.array(keypoints).reshape(-1, 3) # array ([[x1,y1,conf],[x2,y2,conf],...])
#     valid = keypoints[:, 2] > 0.1  # confidenceでフィルタ
#     if not np.any(valid):
#         return None
#     center = keypoints[valid, :2].mean(axis=0)
#     return center

def extract_pose_center(pose): ## 試験的 Body_25 前提
    keypoints = pose.get("pose_keypoints_2d", [])  # OpenPose形式前提
    if not keypoints or len(keypoints) < 3:
        return None
    keypoints = np.array(keypoints).reshape(-1, 3)

    neck = keypoints[1] # Body_25 前提
    if neck[2] > 0.1:
        return neck[:2]
    else:
        return None


def get_point(pose, index):
    kp = pose.get("pose_keypoints_2d", [])
    if not kp or len(kp) < 3 * (index + 1):
        return None
    x, y, conf = kp[3*index:3*index+3]
    return np.array([x, y]) if conf > 0.1 else None

def compute_scale_old(input_pose, ref_pose, idx1=2, idx2=5):
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
        raise ValueError("片方の画像が小さすぎます")
        # return 1.0
    return len_input / len_ref



def compute_scale(input_pose, ref_pose, idx1=2, idx2=5):
    """例: idx1=2(R-Shoulder), idx2=5(L-Shoulder)

    型の欠損に頑健な設計に改造した
    片方の肩が欠損している場合
    - 右肩が欠損している場合 : 左肩のスケールのみを使用して結果を返す
    - 左肩が欠損している場合 : 右肩のスケールのみを使用して結果を返す
    - 両肩が欠損している場合 : デフォルト値 `1.0` を返す
  ]"""
    # Neckと右肩の距離を計算
    neck_idx = 1
    r_shoulder_idx = idx1
    l_shoulder_idx = idx2

    p_neck_input = get_point(input_pose, neck_idx)
    p_r_shoulder_input = get_point(input_pose, r_shoulder_idx)
    p_neck_ref = get_point(ref_pose, neck_idx)
    p_r_shoulder_ref = get_point(ref_pose, r_shoulder_idx)

    scale_r_shoulder = None
    if p_neck_input is not None and p_r_shoulder_input is not None and p_neck_ref is not None and p_r_shoulder_ref is not None:
        len_input_r = np.linalg.norm(p_neck_input - p_r_shoulder_input)
        len_ref_r = np.linalg.norm(p_neck_ref - p_r_shoulder_ref)
        if len_ref_r > 1e-5:  
            scale_r_shoulder = len_input_r / len_ref_r

    # Neckと左肩の距離を計算
    p_l_shoulder_input = get_point(input_pose, l_shoulder_idx)
    p_l_shoulder_ref = get_point(ref_pose, l_shoulder_idx)

    scale_l_shoulder = None
    if p_neck_input is not None and p_l_shoulder_input is not None and p_neck_ref is not None and p_l_shoulder_ref is not None:
        len_input_l = np.linalg.norm(p_neck_input - p_l_shoulder_input)
        len_ref_l = np.linalg.norm(p_neck_ref - p_l_shoulder_ref)
        if len_ref_l > 1e-5: 
            scale_l_shoulder = len_input_l / len_ref_l

    # calc average scale
    scales = [s for s in [scale_r_shoulder, scale_l_shoulder] if s is not None]
    if len(scales) == 0:
        raise ValueError("画像が欠損しているまたは、片方の画像が小さすぎます")
    return sum(scales) / len(scales)

def scale_with_anchor(
    img_tensor, pose_keypoints, scale_factor,
    ref_anchor_target=None, anchor_index=1,
    input_image_shape=None  # ★追加
        ):
    """
    neck を anchor にしてスケーリングし、任意の位置に合わせて平行移動する。
    ref_anchor_target を指定すると、そこに合わせる。
    """
    C, H, W = img_tensor.shape
    img_pil = TF.to_pil_image(img_tensor)

    keypoints = np.array(pose_keypoints).reshape(-1, 3)
    anchor = keypoints[anchor_index][:2]

    new_size = (int(W * scale_factor), int(H * scale_factor))
    img_scaled = img_pil.resize(new_size, Image.BICUBIC)
    img_scaled_tensor = TF.to_tensor(img_scaled)
    new_anchor = anchor * scale_factor

    shift_x = int(round(ref_anchor_target[0] - new_anchor[0])) if ref_anchor_target is not None else int(round(anchor[0] - new_anchor[0]))
    shift_y = int(round(ref_anchor_target[1] - new_anchor[1])) if ref_anchor_target is not None else int(round(anchor[1] - new_anchor[1]))

    if input_image_shape is not None:
        input_H, input_W = input_image_shape
        shift_y += (2*(H - input_H)) ##### ここ本当にあってるか確認が必要 2 倍ずれてるはず
    

    print(f"anchor={anchor}, new_anchor={new_anchor}, ref_anchor_target={ref_anchor_target}, shift_y={shift_y}")

    # breakpoint()

    canvas = torch.zeros((C, H, W), dtype=img_tensor.dtype)

    paste_x0 = max(0, shift_x)
    paste_y0 = max(0, shift_y)
    paste_x1 = min(W, new_size[0] + shift_x)
    paste_y1 = min(H, new_size[1] + shift_y)

    crop_x0 = max(0, -shift_x)
    crop_y0 = max(0, -shift_y)
    crop_x1 = crop_x0 + (paste_x1 - paste_x0)
    crop_y1 = crop_y0 + (paste_y1 - paste_y0)

    canvas[:, paste_y0:paste_y1, paste_x0:paste_x1] = img_scaled_tensor[:, crop_y0:crop_y1, crop_x0:crop_x1]
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

    def align(self, input_keypoints, input_image, reference_keypoints, reference_images, adjust_scale=True):
        input_pose = input_keypoints[0]['people'][0]
        ref_pose_0 = reference_keypoints[0]['people'][0]

        # 入力画像のサイズと neck の座標を取得
        if input_image[0].ndim == 3 and input_image[0].shape[0] > 4:
            input_img = input_image[0].permute(2, 0, 1)
        else:
            input_img = input_image[0]

        _, target_H, target_W = input_img.shape
        input_center = extract_pose_center(input_pose)
        ref_center = extract_pose_center(ref_pose_0)

        if input_center is None or ref_center is None:
            raise ValueError("中心点の抽出に失敗しました")

        # スケール係数を計算
        scale_factor = 1.0
        if adjust_scale:
            scale_factor = compute_scale(input_pose, ref_pose_0, idx1=2, idx2=5)

        aligned_images = []
        for img in reference_images:
            if img.ndim == 3 and img.shape[0] > 4:
                img = img.permute(2, 0, 1)
            elif img.ndim != 3 or img.shape[0] != 3:
                raise ValueError(f"Unexpected image shape: {img.shape}")

            # neck を input_image の neck に揃えるようにスケーリング＋平行移動
            ## TODO おかしいよおおおおおおおおおおおおおおおおおおおおおおおおなんで x 方向ずれるんだよ。。。
            img_aligned = scale_with_anchor(
                    img,
                    ref_pose_0["pose_keypoints_2d"],
                    scale_factor,
                    ref_anchor_target=input_center,
                    input_image_shape=(target_H, target_W)
                )

            # 画像を中央に貼る（キャンバスは input_image と同じサイズ）
            C, H, W = img_aligned.shape
            shifted = torch.zeros((C, target_H, target_W), dtype=img_aligned.dtype)

            offset_x = (target_W - W) // 2
            offset_y = (target_H - H) // 2

            paste_x0 = max(0, offset_x)
            paste_y0 = max(0, offset_y)
            paste_x1 = paste_x0 + min(W, target_W)
            paste_y1 = paste_y0 + min(H, target_H)

            shifted[:, paste_y0:paste_y1, paste_x0:paste_x1] = img_aligned[:, 0:(paste_y1 - paste_y0), 0:(paste_x1 - paste_x0)]
            aligned_images.append(shifted)

        video_tensor = torch.stack(aligned_images, dim=0)
        aligned_video = video_tensor.permute(0, 2, 3, 1)
        return (aligned_video,)




NODE_CLASS_MAPPINGS = {
    "Rebuilt_Video": RebuiltVideo,
    "BoneImageTemporalFixer": BoneImageTemporalFixer,
    "AlignPOSE_KEYPOINTToReference": AlignPOSE_KEYPOINTToReference,
}



