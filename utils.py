import cv2
from pathlib import Path
from matplotlib import pyplot as plt
import subprocess
import pandas as pd
import numpy as np
from pycocotools.coco import COCO
from matplotlib.path import Path as PathPlt
from sklearn.model_selection import train_test_split
from shutil import copy2


# dict of categories
category_dict = {0: "other_stuff", 1: "road", 2: "sidewalk", 3: "greens", 4: "other", 5: "Biker",
                6: "Pedestrian", 7: "Skater", 8: "Cart", 9: "Car", 10: "Bus"}

category_clr = {0: [0, 0, 0], 1: [128, 64, 128], 2: [130, 76, 0], 3: [107, 142, 35],
                4: [0, 0, 0], 5: [255, 22, 96], 6: [102, 51, 0], 7: [9, 143, 150],
                8: [119, 11, 32], 9: [112, 150, 146], 10: [48, 41, 30]}


class VideoDataset:
    def __init__(self, data_root="/media/hdd/stanford_campus_dataset"):
        self.data_root = data_root
        # 1. Get available scenes
        scenes_path = Path(data_root + "/videos").glob('./*')
        self.scenes = {scene.name: {"path": scene, "videos": {}} for scene in scenes_path}

        self.annotations = {}
        self.coco_stuff = {}
        self.masks_stuff = {}
        self.masks_base = {}
        self.references = {}

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
        if len(catIds) == 0:
            return []

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
        print(f"Number of annotations: {len(anns)}")
        return anns

    def create_color_mask(self, scene_name, video_name, idx_frame, show=False):
        mask_dict = {}
        mask_base, _ = self.__check_load_mask_base(scene_name, video_name)
        mask_dict[0] = mask_base.copy()

        mask_stuff = self.__check_load_mask_stuff(scene_name, video_name)

        mask_dict[1] = mask_stuff[1]
        mask_dict[2] = mask_stuff[2]
        mask_dict[3] = mask_stuff[3]
        mask_dict[4] = mask_stuff[4]

        ann_df = self.__check_load_annotations(scene_name, video_name)
        mask_dict[5], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Biker")
        mask_dict[6], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Pedestrian")
        mask_dict[7], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Skater")
        mask_dict[8], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Cart")
        mask_dict[9], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Car")
        mask_dict[10], _ = create_mask_from_bbox(ann_df, idx_frame, mask_base.shape, label="Bus")

        mask_out = mask_base.copy()
        mask_out = np.dstack([mask_out] * 3)
        for cat_id in range(1, max(list(category_dict.keys())) + 1):
            mask_cur = mask_dict[cat_id]
            # plt.imshow(mask_cur)
            # plt.show()
            mask_bool = mask_cur.astype(bool)
            clr = get_clr(category_clr, cat_id)
            mask_out[mask_bool, 0] = clr[0]
            mask_out[mask_bool, 1] = clr[1]
            mask_out[mask_bool, 2] = clr[2]
        if show:
            plt.imshow(mask_out.astype('uint8'))
            plt.show()
        return mask_out

    def show_mask(self, scene_name, video_name, category_id):
        mask_stuff = self.__check_load_mask_stuff(scene_name, video_name)
        plt.imshow(mask_stuff[category_id])
        plt.show()

    def create_color_masks(self, scene_name, video_name, idx_frame_from=0):
        ann_df = self.__check_load_annotations(scene_name, video_name)
        idx_frame_min = ann_df["frame"].min()
        idx_frame_max = ann_df["frame"].max()
        path = Path(self.data_root) / "masks" / scene_name / video_name
        path.mkdir(parents=True, exist_ok=False)
        idx_frame_min = max(idx_frame_min, idx_frame_from)
        for i in range(idx_frame_min, idx_frame_max+1):
            print(f"process frame: {i}")
            mask = self.create_color_mask(scene_name, video_name, i, False)
            cv2.imwrite(str(path / ("frame_" + str(i).zfill(6) + ".png")), mask.astype('uint8'))

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

    def __check_load_mask_stuff(self, scene_name, video_name):
        mask_base, _ = self.__check_load_mask_base(scene_name, video_name)
        if scene_name not in self.masks_stuff:
            self.masks_stuff[scene_name] = {}
        if video_name not in self.masks_stuff[scene_name]:
            self.masks_stuff[scene_name][video_name] = {}
            anns_road = self.get_polygons(scene_name, video_name, ['road'])
            self.masks_stuff[scene_name][video_name][1], _ = create_bin_mask_from_polygons(anns_road, mask_base,
                                                                                           show=False)
            anns_sidewalk = self.get_polygons(scene_name, video_name, ['sidewalk'])
            self.masks_stuff[scene_name][video_name][2], _ = create_bin_mask_from_polygons(anns_sidewalk, mask_base,
                                                                                           show=False)
            anns_greens = self.get_polygons(scene_name, video_name, ['greens'])
            self.masks_stuff[scene_name][video_name][3], _ = create_bin_mask_from_polygons(anns_greens, mask_base,
                                                                                           show=False)
            anns_other = self.get_polygons(scene_name, video_name, ['other'])
            self.masks_stuff[scene_name][video_name][4], _ = create_bin_mask_from_polygons(anns_other, mask_base,
                                                                                           show=False)

        return self.masks_stuff[scene_name][video_name]

    def __check_load_mask_base(self, scene_name, video_name):
        if scene_name not in self.masks_base:
            self.masks_base[scene_name] = {}
        if scene_name not in self.references:
            self.references[scene_name] = {}
        if video_name not in self.references[scene_name]:
            path = Path(self.data_root) / "annotations" / scene_name / video_name / "reference.jpg"
            self.references[scene_name][video_name] = cv2.imread(str(path))
        if video_name not in self.masks_base[scene_name]:
            self.masks_base[scene_name][video_name] = np.zeros(self.references[scene_name][video_name][:, :, 0].shape)
        return self.masks_base[scene_name][video_name], self.references[scene_name][video_name]

    def split_dataset(self, parts_size=[0.8, 0.0, 0.2], out_path="data"):
        """
        Splits all dataset into train/val/test parts
        :param
            parts_size: [train_size, val_size, test_size]
            out_path: output directory name, relative to self.data_root
        :return:

        """

        # split every directory separately
        # get list of all files in directory recursive
        scenes_paths = (Path(self.data_root) / "masks").glob("./*")
        scenes_paths = [v for v in scenes_paths]
        for scene_path in scenes_paths:
            # get list of videos
            videos_paths = (Path(self.data_root) / "masks" / scene_path.name).glob("./*")
            videos_paths = [v for v in videos_paths]
            scene_name = scene_path.name

            for video_path in videos_paths:
                video_name = video_path.name
                print(f"Processing scene: {scene_name}, video_name: {video_name}")
                files = list(video_path.rglob("*.png"))
                print(f"Number of files: {len(files)}")

                train_size, val_size, test_size = parts_size
                x_train, x_remain = train_test_split(files, test_size=(val_size + test_size))

                new_test_size = np.around(test_size / (val_size + test_size), 2)
                print(f"New test size: {new_test_size}")
                # To preserve (new_test_size + new_val_size) = 1.0
                new_val_size = 1.0 - new_test_size

                if new_test_size == 1.0:
                    x_val = []
                    x_test = x_remain
                elif new_test_size == 0.0:
                    x_val = x_remain
                    x_test = []
                else:
                    x_val, x_test = train_test_split(x_remain, test_size=new_test_size)
                print(f"Train: {len(x_train)}, val: {len(x_val)}, test: {len(x_test)}")

                out_p = Path(self.data_root) / out_path
                out_p.mkdir(parents=True, exist_ok=True)

                # train dataset
                out_p_train = out_p / "Train"
                (out_p_train / "Mask").mkdir(parents=True, exist_ok=True)
                (out_p_train / "Image").mkdir(parents=True, exist_ok=True)
                for file_p in x_train:

                    frame_idx = int(file_p.stem.split("_")[1])
                    copy2(str(file_p), str(out_p_train / "Mask" / (scene_name + "_" + video_name + "_" + file_p.name)))
                    copy2(str(Path(self.data_root) / "frames" / scene_name / video_name /
                                   ("frame_" + str(frame_idx+1).zfill(6) + ".jpg")),
                          str(out_p_train / "Image" / (scene_name + "_" + video_name + "_" + file_p.stem + ".jpg")))
                print("Finished copy train images")

                # validation dataset
                out_p_val = out_p / "Val"
                (out_p_val / "Mask").mkdir(parents=True, exist_ok=True)
                (out_p_val / "Image").mkdir(parents=True, exist_ok=True)
                for file_p in x_val:
                    frame_idx = int(file_p.stem.split("_")[1])
                    copy2(str(file_p), str(out_p_val / "Mask" / (scene_name + "_" + video_name + "_" + file_p.name)))
                    copy2(str(Path(self.data_root) / "frames" / scene_name / video_name /
                                   ("frame_" + str(frame_idx+1).zfill(6) + ".jpg")),
                          str(out_p_val / "Image" / (scene_name + "_" + video_name + "_" + file_p.stem + ".jpg")))
                print("Finished copy validation images")

                # test dataset
                out_p_test = out_p / "Test"
                (out_p_test / "Mask").mkdir(parents=True, exist_ok=True)
                (out_p_test / "Image").mkdir(parents=True, exist_ok=True)
                for file_p in x_test:
                    frame_idx = int(file_p.stem.split("_")[1])
                    copy2(str(file_p), str(out_p_test / "Mask" / (scene_name + "_" + video_name + "_" + file_p.name)))
                    copy2(str(Path(self.data_root) / "frames" / scene_name / video_name /
                                   ("frame_" + str(frame_idx + 1).zfill(6) + ".jpg")),
                          str(out_p_test / "Image" / (scene_name + "_" + video_name + "_" + file_p.stem + ".jpg")))
                print("Finished copy test images")


def create_bin_mask_from_polygons(annotations, mask_base, show=False):
    """ Create mask from annotations. Consider all item in annotations are the same category
        Args:
            annotations (list): list of annotation items for same category
            mask_base (np.ndarray): input mask filled with zeros
            show (bool): show or not binary mask in GUI window
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


def create_mask_from_bbox(df, idx_frame, mask_shape, label=None, show=False):
    mask_out = np.zeros(mask_shape)
    masks = []
    if label is None:
        data = df[(df["frame"] == idx_frame) & (df["lost"] == 0) & (df["occluded"] == 0)].copy()
    else:
        data = df[(df["frame"] == idx_frame) & (df["label"] == label) & (df["lost"] == 0) & (df["occluded"] == 0)].copy()
    # print(data)
    for index, row in data.iterrows():
        h, w = mask_shape

        ymin = row.ymin
        xmin = row.xmin
        ymax = row.ymax
        xmax = row.xmax
        points = np.array([ymin, xmin, ymax, xmin, ymax, xmax, ymin, xmax]).reshape((4, 2))

        poly_path = PathPlt(points)

        x, y = np.mgrid[:h, :w]
        coors = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1)))  # coors.shape is (width*height,2)

        mask_new = poly_path.contains_points(coors).reshape(mask_shape)
        mask_out = np.logical_or(mask_out, mask_new)
        masks.append(mask_new)
    if show:
        plt.imshow(mask_out)
        plt.show()

    return mask_out, masks


def get_label_id(cat_dict, label_name):
    for k, v in cat_dict.items():
        if v == label_name:
            return k
    return -1


def get_clr(cat_clr, label_id):
    for k, v in cat_clr.items():
        if k == label_id:
            return v
    return None
