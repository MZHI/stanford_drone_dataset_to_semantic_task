#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from utils import VideoDataset, create_bin_mask_from_polygons

# 1. load videos
v_dataset = VideoDataset()
print("Scenes: ".format(v_dataset.get_scenes()))

scene_name = "deathCircle"
video_name = "video1"
first_frame = v_dataset.get_frame(scene_name, video_name)

# 2. load annotations
ann_df = v_dataset.load_annotations(scene_name, video_name)
print(ann_df.head())

# 3. Unique classes
classes = ann_df["label"].unique()
print(f"Classes: {classes}")

# 4. Load polygons for specific category
filter_classes = ["greens"]
anns = v_dataset.get_polygons(scene_name, video_name, filter_classes)
print(f"{len(anns)} annotations for class {filter_classes} loaded")

# 5. Create mask from polygons for specific category
# mask_base = np.zeros(first_frame.shape[0: 2])
# mask_out, masks = create_bin_mask_from_polygons(anns, mask_base, True)

# v_dataset.show_mask(scene_name, video_name, 1)

# clr_mask = v_dataset.create_color_mask(scene_name, video_name, 6227, show=True)
# v_dataset.create_color_masks(scene_name, video_name, idx_frame_from=0)

# 6. Split dataset to train/val/test parts
v_dataset.split_dataset(parts_size=[0.7, 0.2, 0.1])
