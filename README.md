# stanford_drone_dataset_to_semantic_task
Task: get dataset [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) and study network for semantic segmentation not only for moving labeled objects ['Biker' 'Pedestrian' 'Skater' 'Cart' 'Car' 'Bus'], but also for background categories so that such segmentation can be used to navigate a mobile robot 

# Stages of solution: 
1. Background categories were selected: "road", "sidewalk", "greens", "other_stuff"
2. For each video sequence get reference frame and label it using some tool. I used [coco annotator tool](https://github.com/jsbroks/coco-annotator) for labeling this frames and save results to coco format. Example of labeling for class `sidewalk`: 

image | mask 
------|------
![img](/images/reference.jpg) | ![mask](/images/mask.png) 

3. Merge annotations from two domains: one from original stanford dataset, and another from my labeling. The only two sequences were labeled: deathCircle->video1 and bookstore->video0. The result of merging is creating colored masks, where categories have next priority (from lowest to highest): ['other_stuff'] -> ['road'] -> ['sidewalk'] -> ['greens'] -> ['Biker'|'Pedestrian'|'Skater'|'Cart'|'Car'|'Bus']. Result: 

image | mask 
------|------
![img](/images/bookstore_video0_frame_005687.jpg) | ![mask](/images/bookstore_video0_frame_005687.png) 

4. Use [Segmentation models pytorch repo](https://github.com/qubvel/segmentation_models.pytorch) for study U-net network using transfer learning (using pretrained on ImageNet dataset weights)

# How to use
1. For working with video from [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) use `VideoDataset` class from `utils.py`:
```python
from utils import VideoDataset
v_dataset = VideoDataset(data_root='path to data')
```
1.1. Show all available scenes:
```python
print("Scenes: ".format(v_dataset.get_scenes()))
```
1.2. Get first frame from specific video
```python
scene_name = "deathCircle"
video_name = "video1"
first_frame = v_dataset.get_frame(scene_name, video_name)
```
1.3. Get last frame from specific video
```python
scene_name = "deathCircle"
video_name = "video1"
last_frame = v_dataset.get_frame(scene_name, video_name, is_last=True)
```
1.4. Split video sequence into frames for specific scene and video
```python
v_dataset.split_video(scene_name, video_name, destination_root='destination path')
```
2. Creating color masks. For creating color masks for specific `scene` and `video` move corresponding `stuff.json` from directory `./background_categories_annotations/scene/video/stuff.json` to data directory into `./annotations/scene/video/stuff.json` and then use next code:
```python
v_dataset.create_color_masks(scene_name, video_name, idx_frame_from=0)
```
3. Split dataset into train/val/test:
```python
v_dataset.split_dataset(parts_size=[0.7, 0.2, 0.1], out_path='output path')
```
4. Use notebook `transfer_learning_unet.ipynb` for transfer learning. This notebook based on next example from [segmentation_models.pytorch repo](https://github.com/qubvel/segmentation_models.pytorch/blob/master/examples/cars%20segmentation%20(camvid).ipynb)

