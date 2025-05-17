import torch
import torch.nn.functional as F
import numpy as np
import cv2
from torchvision.transforms import functional as TF
from PIL import Image
import random

# 完全な BODY_25 対応（顔周辺含む）
# OpenPose BODY_25 準拠の接続リスト
# https://github.com/lllyasviel/ControlNet/discussions/266
BODY_EDGES = [
    (1, 2), (1, 5), (2, 3), (3, 4),
    (5, 6), (6, 7),
    (1, 8), (8, 9), (9, 10),
    (1, 11), (11, 12), (12, 13),
    (1, 0), (0, 14), (14, 16),
    (0, 15), (15, 17)
]

BODY_COLORS = [
    (153, 0, 0),     # 1–2 Right Shoulderblade
    (153, 51, 0),    # 1–5 Left Shoulderblade
    (153, 102, 0),   # 2–3 Right Arm
    (153, 153, 0),   # 3–4 Right Forearm
    (102, 153, 0),   # 5–6 Left Arm
    (51, 153, 0),    # 6–7 Left Forearm
    (0, 153, 0),     # 1–8 Right Torso
    (0, 153, 51),    # 8–9 Right Upper Leg
    (0, 153, 102),   # 9–10 Right Lower Leg
    (0, 153, 153),   # 1–11 Left Torso
    (0, 102, 153),   # 11–12 Left Upper Leg
    (0, 51, 153),    # 12–13 Left Lower Leg
    (0, 0, 153),     # 1–0 Head
    (51, 0, 153),    # 0–14 R-Eyebrow
    (102, 0, 153),   # 14–16 R-Ear
    (153, 0, 153),   # 0–15 L-Eyebrow
    (153, 0, 102)    # 15–17 L-Ear
]

BODY_JOINT_COLORS = [
    (255,   0,   0),    # 0 - nose
    (255,  85,   0),    # 1 - neck
    (255, 170,   0),    # 2 - R shoulder
    (255, 255,   0),    # 3 - R elbow
    (170, 255,   0),    # 4 - R wrist
    (85,  255,   0),    # 5 - L shoulder
    (0,   255,   0),    # 6 - L elbow
    (0,   255,  85),    # 7 - L wrist
    (0,   255, 170),    # 8 - R hip
    (0,   255, 255),    # 9 - R knee
    (0,   170, 255),    # 10 - R ankle
    (0,    85, 255),    # 11 - L hip
    (0,     0, 255),    # 12 - L knee
    (85,    0, 255),    # 13 - L ankle
    (255,   0, 255),    # 14 - R eye
    (255,   0, 255),    # 15 - L eye
    (255,   0, 170),    # 16 - R ear
    (255,   0,  85),    # 17 - L ear
]


HAND_EDGES = [(i, i + 1) for i in range(0, 20) if (i + 1) % 4 != 0] + [(0, i) for i in [1, 5, 9, 13, 17]]

HAND_COLORS = {
    0: (255, 0, 0),
    1: (255, 64, 0),
    2: (255, 128, 0),
    3: (255, 192, 0),
    4: (255, 255, 0),
    5: (192, 255, 0),
    6: (128, 255, 0),
    7: (64, 255, 0),
    8: (0, 255, 0),
    9: (0, 255, 64),
    10: (0, 255, 128),
    11: (0, 255, 192),
    12: (0, 255, 255),
    13: (0, 192, 255),
    14: (0, 128, 255),
    15: (0, 64, 255),
    16: (0, 0, 255),
    17: (64, 0, 255),
    18: (128, 0, 255),
    19: (192, 0, 255),
    20: (255, 0, 255),
}


FACE_CIRCLE_SIZE = 5
BODY_CIRCLE_SIZE = 10
HAND_CIRCLE_SIZE = 5



def get_point(pose, index, key_name="pose_keypoints_2d"):
    """
    __get__ function
    get all keypoints from keypoints dict
    """
    kp = pose.get(key_name, [])
    if not kp or len(kp) < 3 * (index + 1):
        return None
    x, y, conf = kp[3 * index:3 * index + 3]
    return np.array([x, y]) if conf > 0.1 else None


def extract_pose_center(pose):
    """
    calcurate pose center
    Use neck
    """
    keypoints = pose.get("pose_keypoints_2d", [])
    if not keypoints or len(keypoints) < 3:
        return None
    keypoints = np.array(keypoints).reshape(-1, 3)
    neck = keypoints[1]
    return neck[:2] if neck[2] > 0.1 else None


def compute_scale(input_pose, control_pose, idx1=2, idx2=5):
    """
    calcurate sacele 
    The scale is automatically calculated using the shoulder width at each end of the neck.
    """
    neck_idx = 1

    def length(p1, p2):
        return np.linalg.norm(p1 - p2)

    def safe_scale(p_neck_input, p_shoulder_input, p_neck_control, p_shoulder_control):
        if any(p is None for p in [p_neck_input, p_shoulder_input, p_neck_control, p_shoulder_control]):
            return None
        len_input = length(p_neck_input, p_shoulder_input)
        len_control = length(p_neck_control, p_shoulder_control)
        return len_input / len_control if len_control > 1e-5 else None

    p_neck_input = get_point(input_pose, neck_idx)
    p_neck_control = get_point(control_pose, neck_idx)

    scale_r = safe_scale(p_neck_input, get_point(input_pose, idx1), p_neck_control, get_point(control_pose, idx1))
    scale_l = safe_scale(p_neck_input, get_point(input_pose, idx2), p_neck_control, get_point(control_pose, idx2))

    scales = [s for s in [scale_r, scale_l] if s is not None]
    if not scales:
        raise ValueError("肩または首のキーポイントが欠損しているか、距離が小さすぎます")
    return sum(scales) / len(scales)


def render_pose_only(pose_kps, hand_l_kps=None, hand_r_kps=None, size=(768, 576)):
    """
    Reenderer forbody and hands
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)

    # ======== BODY ==========
    if pose_kps:
        kps = np.array(pose_kps).reshape(-1, 3)
        for idx, (i, j) in enumerate(BODY_EDGES):
            if kps[i][2] > 0.1 and kps[j][2] > 0.1:
                pt1 = kps[i][:2].astype(np.int32)
                pt2 = kps[j][:2].astype(np.int32)
                color = BODY_COLORS[idx % len(BODY_COLORS)]

                center = ((pt1 + pt2) // 2).astype(int)
                length = int(np.linalg.norm(pt1 - pt2))
                angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
                axes = (max(length // 2, 1), 8)

                cv2.ellipse(img, tuple(center), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(pt1), BODY_CIRCLE_SIZE, BODY_JOINT_COLORS[i], -1)
                cv2.circle(img, tuple(pt2), BODY_CIRCLE_SIZE, BODY_JOINT_COLORS[j], -1)

    # ======== HANDS ========== None なら
    def draw_hand(hand_kps, hand_side):
        hand = np.array(hand_kps).reshape(-1, 3)
        for i, j in HAND_EDGES:
            if hand[i][2] > 0.1 and hand[j][2] > 0.1:
                pt1 = hand[i][:2].astype(np.int32)
                pt2 = hand[j][:2].astype(np.int32)

                center = ((pt1 + pt2) // 2).astype(int)
                length = int(np.linalg.norm(pt1 - pt2))
                angle = np.degrees(np.arctan2(pt2[1] - pt1[1], pt2[0] - pt1[0]))
                axes = (max(length // 2, 1), 5)

                # 左右でボーンの色を分ける（例えば）
                color = HAND_COLORS.get(i, (255, 255, 255))  # デフォルト白
                joint_color = (0, 0, 255)  # 全ジョイント青で統一

                cv2.ellipse(img, tuple(center), axes, angle, 0, 360, color, -1, lineType=cv2.LINE_AA)
                cv2.circle(img, tuple(pt1), HAND_CIRCLE_SIZE, joint_color, -1)
                cv2.circle(img, tuple(pt2), HAND_CIRCLE_SIZE, joint_color, -1)

    if hand_l_kps is not None:
        draw_hand(hand_l_kps, "left")
    if hand_r_kps is not None:
        draw_hand(hand_r_kps, "right")

    return TF.to_tensor(Image.fromarray(img))


def render_face_only(face_kps, size=(768, 576)):
    """
    Reenderer for face
    """
    img = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    if face_kps:
        pts = np.array(face_kps).reshape(-1, 3)
        for x, y, c in pts:
            if c > 0.1:
                cv2.circle(img, (int(x), int(y)), FACE_CIRCLE_SIZE, (255, 255, 255), -1)
    return TF.to_tensor(Image.fromarray(img))


def scale_with_anchor(img_tensor, keypoints, scale_factor, control_anchor_target=None, anchor_index=None, return_shift=False, override_shift=None, custom_anchor=None):
    """
    Resize body image with anchor 0 : "neck"
    """
    C, H, W = img_tensor.shape
    keypoints = np.array(keypoints).reshape(-1, 3)

    # アンカーの決定
    if custom_anchor is not None:
        anchor = custom_anchor
    elif anchor_index is not None and keypoints[anchor_index][2] > 0.1:
        anchor = keypoints[anchor_index][:2]
    else:
        return (torch.zeros_like(img_tensor), np.array([0, 0])) if return_shift else torch.zeros_like(img_tensor)

    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    img_scaled_tensor = F.interpolate(img_tensor.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False)[0]

    new_anchor = anchor * scale_factor

    # shift の決定
    if override_shift is not None:
        shift = np.array(override_shift).astype(int)
    elif control_anchor_target is not None:
        shift = (control_anchor_target - new_anchor).astype(int)
    else:
        shift = np.array([0, 0])

    # 貼り付け処理
    canvas = torch.zeros((C, H, W), dtype=img_tensor.dtype)
    paste_x0 = max(0, shift[0])
    paste_y0 = max(0, shift[1])
    crop_x0 = max(0, -shift[0])
    crop_y0 = max(0, -shift[1])
    paste_h = min(H - paste_y0, img_scaled_tensor.shape[1] - crop_y0)
    paste_w = min(W - paste_x0, img_scaled_tensor.shape[2] - crop_x0)

    canvas[:, paste_y0:paste_y0+paste_h, paste_x0:paste_x0+paste_w] = \
        img_scaled_tensor[:, crop_y0:crop_y0+paste_h, crop_x0:crop_x0+paste_w]

    return (canvas, shift) if return_shift else canvas

def custom_scale_around_anchor(tensor, anchor, scale_factor):
    """
    Resize face image with anchor 30 : "nose"
    """
    # アンカーの決定
    C, H, W = tensor.shape
    new_H, new_W = int(H * scale_factor), int(W * scale_factor)
    tensor_scaled = F.interpolate(tensor.unsqueeze(0), size=(new_H, new_W), mode="bicubic", align_corners=False)[0]

    new_anchor = anchor * scale_factor

    # shift の決定
    shift = (anchor - new_anchor).astype(int)

    # 拡大後サイズで canvas を確保（貼り付け切れを防ぐ）
    canvas = torch.zeros((C, H, W), dtype=tensor.dtype)

    paste_x0 = max(0, shift[0])
    paste_y0 = max(0, shift[1])
    crop_x0 = max(0, -shift[0])
    crop_y0 = max(0, -shift[1])
    paste_h = min(H - paste_y0, tensor_scaled.shape[1] - crop_y0)
    paste_w = min(W - paste_x0, tensor_scaled.shape[2] - crop_x0)

    # 貼り付け処理
    if paste_h > 0 and paste_w > 0:
        canvas[:, paste_y0:paste_y0+paste_h, paste_x0:paste_x0+paste_w] = \
            tensor_scaled[:, crop_y0:crop_y0+paste_h, crop_x0:crop_x0+paste_w]

    return canvas




class AlignPOSE_KEYPOINTToReference:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_image": ("IMAGE", {"forceInputList": True}),
                "input_keypoints": ("POSE_KEYPOINT",),
                "control_images": ("IMAGE", {"forceInputList": True}),
                "control_keypoints": ("POSE_KEYPOINT", {"forceInputList": True}),
                "adjust_scale": ("BOOLEAN", {"default": True}),
                "custom_resize_scale": ("BOOLEAN", {"default": False}),
                "custom_scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 100.0}),
                "adjust_face_scale":("BOOLEAN", {"default": True}),
                "face_scaling_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100.0}),
                "use_person_input_image": ("INT", {"default": 1, "min": 1, "max": 10}), # n 番目の人間のポーズを使用する
                "use_person_control_image": ("INT", {"default": 1, "min": 1, "max": 10}), # n 番目の人間のポーズを使用する
                "draw_hands":("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("aligned_video",)
    FUNCTION = "align"
    CATEGORY = "MyPromptTest"

    def align(self, input_keypoints, input_image, control_keypoints, control_images, 
              adjust_scale=True,custom_resize_scale=False, custom_scaling_factor=1.0, 
              adjust_face_scale=True, face_scaling_factor=1.0, use_person_input_image=1,use_person_control_image=1,draw_hands=True):

        if not adjust_scale:
            return (input_image,) 
        
        try:
            input_pose = input_keypoints[0]['people'][use_person_input_image - 1]
        except IndexError:
            raise ValueError(f"input_keypoints の 'people' に {use_person_input_image} 番目の人物が存在しません。")
        
        first_frame_control_pose = None
        for frame_keypoints in control_keypoints:
            people = frame_keypoints.get("people", [])
            if len(people) >= use_person_control_image:
                first_frame_control_pose = people[use_person_control_image - 1]
                break

        if first_frame_control_pose is None:
            raise ValueError("control_keypoints に有効な人物が含まれるフレームが見つかりませんでした。")

        # if input_pose is None :
        #      raise ValueError("input_keypoints に有効な人物が含まれていません。")  
        # elif first_frame_control_pose is None:
        #     raise ValueError("control_keypoints に有効な人物が含まれていません。")

        input_img = input_image[0].permute(2, 0, 1) if input_image[0].ndim == 3 else input_image[0]
        _, target_H, target_W = input_img.shape

        # Extract pose center
        input_center = extract_pose_center(input_pose)
        control_center = extract_pose_center(first_frame_control_pose)
        if input_center is None or control_center is None:
            raise ValueError("中心点の抽出に失敗しました")

        scale_factor = compute_scale(input_pose, first_frame_control_pose) if adjust_scale else 1.0
        if custom_resize_scale:
            scale_factor *= custom_scaling_factor

        aligned_images = []
        for i, control_img in enumerate(control_images):
            people = control_keypoints[i].get('people', [])
            if len(people) < use_person_control_image:
                print(f"[Warning] フレーム {i} に {use_person_control_image} 番目の人物が検出されていません。スキップします。")
                continue

            control_pose = people[use_person_control_image - 1]

            face_key = control_pose.get("face_keypoints_2d", [])
            if draw_hands:
                hand_l_key = control_pose.get("hand_left_keypoints_2d", [])
                hand_r_key = control_pose.get("hand_right_keypoints_2d", [])
            else:
                hand_l_key, hand_r_key = None, None
            pose_key = control_pose.get("pose_keypoints_2d", [])

            # キーポイントが有効でない場合スキップ
            if not pose_key or not face_key:
                print(f"[Warning] フレーム {i} に有効なキーポイントが存在しません。スキップします。")
                continue

            body_tensor_raw = render_pose_only(pose_key, hand_l_key, hand_r_key, size=(target_W, target_H))
            face_tensor_raw = render_face_only(face_key, size=(target_W, target_H))

            try:
                # ==== 体をスケール・シフトで整列 ====
                body_tensor, body_shift = scale_with_anchor(
                    body_tensor_raw, pose_key,
                    scale_factor, control_anchor_target=input_center,
                    anchor_index=1, return_shift=True
                )

                # ==== 顔を体と同じスケール・シフトで整列 ====
                face_tensor_base, _ = scale_with_anchor(
                    face_tensor_raw, face_key,
                    scale_factor, override_shift=body_shift,
                    anchor_index=1, return_shift=True
                )

                # ==== 鼻キーポイントを基準に顔だけスケール ====
                face_anchor = get_point(control_pose, 30, key_name="face_keypoints_2d")
                if adjust_face_scale:
                    if face_anchor is not None:
                        anchor = face_anchor * scale_factor + body_shift
                        face_tensor = custom_scale_around_anchor(face_tensor_base, anchor, face_scaling_factor)
                    else:
                        print(f"[Warning] フレーム {i} に鼻のキーポイントがありません。スキップします。")
                        continue
                else:
                    face_tensor = face_tensor_base

                composed = torch.clamp(body_tensor + face_tensor, 0, 1)
                aligned_images.append(composed)

            except Exception as e:
                print(f"[Warning] フレーム {i} の処理中にエラーが発生しました: {e}")
                continue

        video_tensor = torch.stack(aligned_images, dim=0).permute(0, 2, 3, 1)
        return (video_tensor,)



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

class RebuiltVideo:
    """
    ポスタリゼーション時間と同様の効果
    各フレームを n 毎複製することで n 倍の長さの動画（Tensor）にするコード
    """
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

class MakeBlackoutFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_tensor": ("IMAGE",),
                "blackout_ratio": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0}),
                "num_blocks": ("INT", {"default": 3, "min": 1, "max": 100}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "blackout"

    CATEGORY = "custom/video"

    def blackout(self, video_tensor, blackout_ratio, num_blocks):
        """
        video_tensor: torch.Tensor of shape (T, H, W, C) with float values in [0, 1]
        """

        T, H, W, C = video_tensor.shape
        total_blackout_frames = int((T - 1) * blackout_ratio)
        block_length = max(1, total_blackout_frames // num_blocks)

        candidate_indices = list(range(1, T))  # keep frame 0 visible if desired
        blackout_indices = set()
        tried_starts = set()

        while len(blackout_indices) < total_blackout_frames and len(tried_starts) < len(candidate_indices):
            start = random.choice(candidate_indices)
            if start in tried_starts:
                continue
            tried_starts.add(start)
            block = list(range(start, min(start + block_length, T)))
            blackout_indices.update(block)

        blackout_tensor = video_tensor.clone()
        for i in blackout_indices:
            blackout_tensor[i] = torch.zeros((H, W, C), dtype=video_tensor.dtype, device=video_tensor.device)

        return (blackout_tensor,)

NODE_CLASS_MAPPINGS = {
    "Rebuilt_Video": RebuiltVideo,
    "BoneImageTemporalFixer": BoneImageTemporalFixer,
    "AlignPOSE_KEYPOINTToReference": AlignPOSE_KEYPOINTToReference,
    "MakeBlackoutFrame": MakeBlackoutFrame,
}



