#!/usr/bin/env python3

from pathlib import Path
import numpy as np
from utils import VideoDataset, create_mask_from_polygons

# 1. load videos
v_dataset = VideoDataset()
print("Scenes: ".format(v_dataset.get_scenes()))

scene_name = "bookstore"
video = "video0"
first_frame = v_dataset.get_frame(scene_name, video)

# 2. load annotations
ann_df = v_dataset.load_annotations(scene_name, video)
print(ann_df.head())

# 3. Unique classes
classes = ann_df["label"].unique()
print(f"Classes: {classes}")

# 4. Load polygons for specific category
filter_classes = ["greens"]
anns = v_dataset.get_polygons(scene_name, video, filter_classes)
print(f"{len(anns)} annotations for class {filter_classes} loaded")

# 5. Create mask from polygons for specific category
mask_base = np.zeros(first_frame.shape[0: 2])
mask_out, masks = create_mask_from_polygons(anns, mask_base, True)


