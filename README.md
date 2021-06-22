# semantic_stanford_campus_test
Task: get dataset [Stanford Drone Dataset](https://cvgl.stanford.edu/projects/uav_data/) and study network for semantic segmentation not only for moving labeled objects ['Biker' 'Pedestrian' 'Skater' 'Cart' 'Car' 'Bus'], but also for background categories so that such segmentation can be used to navigate a mobile robot 

Stages of solution: 
1. Select background categories: "road", "sidewalk", "greens", "other_stuff"
2. For each video sequence get reference frame and label using some tool. I used [coco annotator tool](https://github.com/jsbroks/coco-annotator) for labeling this frames and save results to coco format
3. Merge annotations from two domains: one from original stanford dataset, and another from my labeling. The only two sequences were labeled: deathCircle->video1 and bookstore->video0. The result of merging is creating colored masks, where categories have next priority (from lowest to highest): ['other_stuff'] -> ['road'] -> ['sidewalk'] -> ['greens'] -> ['Biker'|'Pedestrian'|'Skater'|'Cart'|'Car'|'Bus']
4. Use [Segmentation models pytorch repo](https://github.com/qubvel/segmentation_models.pytorch) for study U-net network using transfer learning (using pretrained on ImageNet dataset weights)
