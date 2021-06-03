import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from matplotlib.path import Path as PathPlt


class VideoDataset:
    def __init__(self, data_root="/media/hdd/stanford_campus_dataset"):
        self.data_root = data_root
        # 1. Get available scenes
        scenes_path = Path(data_root + "/videos").glob('./*')
        self.scenes = {scene.name: {"path": scene, "videos": {}} for scene in scenes_path}

        self.annotations = {}
        self.coco_stuff = {}

        # 2. For each scene, get list of available videos
        for scene_name, scene in self.scenes.items():
            videos = scene["path"].glob("./*")
            for video_path in videos:
                scene["videos"][video_path.name] = video_path / "video.mov"

    def get_scenes(self):
        return [k for k, v in self.scenes.items()]

    def get_frame(self, scene_name, video_name, is_last=False):
        cap = cv2.VideoCapture(str(self.scenes[scene_name]["videos"][video_name]))
        if is_last:
            cap.set(cv2.CAP_PROP_POS_FRAMES, cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1)
        success, frame = cap.read()
        if not success:
            print(f"Can't read first frame for scene '{scene_name}' and video '{video_name}'")
        cap.release()
        return frame

    def split_video(self, scene_name, video_name, destination_root='/media/hdd/stanford_campus_dataset/frames'):
        d_root = Path(destination_root)
        destination_path = d_root / scene_name / video_name
        destination_path.mkdir(parents=True, exist_ok=True)

        # ffmpeg -i video.webm thumb%04d.jpg -hide_banner

        command = ['ffmpeg',
                   '-i', str(self.scenes[scene_name]["videos"][video_name]),
                   str(destination_path / "frame_%6d.jpg"),
                   '-hide_banner']
        p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=False, bufsize=0)
        out, err = p.communicate()
        print(f"Error: {err}")
        return p

    def load_annotations(self, scene_name, video_name):
        return self.__check_load_annotations(scene_name, video_name)

    def load_annotations_stuff(self, scene_name, video_name):
        coco = self.__check_load_annotations_stuff(scene_name, video_name)
        catIDs = coco.getCatIds()
        cats = coco.loadCats(catIDs)
        return cats, catIDs

    def get_polygons(self, scene_name, video_name, filter_classes):
        coco = self.__check_load_annotations_stuff(scene_name, video_name)
        catIds = coco.getCatIds(catNms=filter_classes)

        # Get image containing the above Category IDs
        imgIds = coco.getImgIds(catIds=catIds)
        img = coco.loadImgs(imgIds[0])[0]
        #     I = io.imread('{}/annotations/{}/{}/{}'.format(root_dir,scene_name,video, img['file_name']))/255.0
        #     plt.axis('off')
        #     plt.imshow(I)
        #     plt.show()

        # Load and display instance annotations
        #     plt.imshow(I)
        #     plt.axis('off')
        annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco.loadAnns(annIds)
        return anns

    def __check_load_annotations(self, scene_name, video_name):
        ann_path = Path(self.data_root) / "annotations" / scene_name / video_name / "annotations.txt"
        if not ann_path.is_file:
            raise Exception(f"Path {ann_path} does not exist")
        if scene_name not in self.annotations:
            self.annotations[scene_name] = {}
        if video_name not in self.annotations[scene_name]:
            self.annotations[scene_name][video_name] = pd.read_csv(ann_path, names=['track_id', 'xmin', 'ymin', 'xmax',
                                                            'ymax', 'frame', 'lost', 'occluded', 'generated', 'label'],
                                                            header=None, sep=' ')
        return self.annotations[scene_name][video_name]

    def __check_load_annotations_stuff(self, scene_name, video_name):
        ann_path = Path(self.data_root) / "annotations" / scene_name / video_name / "stuff.json"
        if not ann_path.is_file():
            raise FileExistsError()
        if scene_name not in self.coco_stuff:
            self.coco_stuff[scene_name] = {}

        if video_name not in self.coco_stuff[scene_name]:
            self.coco_stuff[scene_name][video_name] = COCO(ann_path)
        return self.coco_stuff[scene_name][video_name]


def create_mask_from_polygons(annotations, mask_base, show=False):
    """ Create mask from annotations. Consider all item in annotations are the same category
        Args:
            annotations (list): list of annotation items for same category
            mask_base (np.ndarray): input mask filled with zeros

        Returns:
            mask_out: Returns mask, which consists of all polygons from annotations, merged into one mask
            masks: list of masks, each correspond to polygon from annotations
        """
    masks = []
    h, w = mask_base.shape
    mask_out = mask_base.copy()
    for ann in annotations:
        if 'segmentation' in ann:
            seg = ann['segmentation'][0]
            points = np.array(seg).reshape((int(len(seg)/2), 2))
            points = np.flip(points, axis=1)
            poly_path = PathPlt(points)

            x, y = np.mgrid[:h, :w]
            coors = np.hstack((x.reshape(-1, 1), y.reshape(-1,1))) # coors.shape is (width*height,2)

            mask = poly_path.contains_points(coors).reshape((h, w))
            masks.append(mask)
            mask_out = np.logical_or(mask_out, mask)
#             if show:
#                 plt.imshow(mask.reshape(mask_shape))
#                 plt.show()
    if show:
        plt.imshow(mask_out)
        plt.show()
    return mask_out, masks




