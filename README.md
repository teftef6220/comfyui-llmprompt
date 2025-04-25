# ComfyUI LLM Promp

## LLM Prompt
Qwen7B を用いた V2T, I2T を自動で行うカスタムノード

Add LLM Save Dir   
```mkdir /mnt/blue1/cho/SD_Models/models/LLM/```

And write it in extra_model_pathes.yaml

Add LLM Save Path in 
    
```folder_pathes.py```  
    
    folder_names_and_paths["LLM"] = ([os.path.join(models_dir, "LLM")], supported_pt_extensions)


## AlignPOSE_KEYPOINTToReference

This module is a custom ComfyUI node that uses pose_keypoints_2d data obtained from OpenPose (or DWPose) to align reference images to the posture of an input image, specifically by scaling and shifting based on the neck position.  
The pose keypoint information of the reference image is used to align it to the input image.

- Supports the Body_25 format (OpenPose keypoint format)

- Scales the reference image based on the neck keypoint as the anchor.

### Demo

Origin Reference Pose ↓↓  
![Result](assets/slow_pose_gunndum_pose_centercrop.gif)  

Input Image Pose (Target) ↓↓
![Result](assets/Input_Image_Pose.png)  

Use These Two Pose Image and Key Point to Align.

Aligned Image Pose ↓↓
![Result](assets/Fixed_pose.gif)
...


