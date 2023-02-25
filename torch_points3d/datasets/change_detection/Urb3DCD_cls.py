import os
import os.path as osp
from itertools import repeat, product
import numpy as np
import h5py
import torch
import random
import glob
import csv
from plyfile import PlyData, PlyElement
from torch_geometric.data import Data, extract_zip, Dataset
from torch_geometric.data.dataset import files_exist
from torch_geometric.data import DataLoader
import torch_geometric.transforms as T
import logging
from sklearn.neighbors import NearestNeighbors, KDTree
from tqdm.auto import tqdm as tq
import csv
import pandas as pd
import pickle
import gdown
import shutil

from torch_points3d.core.data_transform import GridSampling3D, CylinderSampling, SphereSampling
from torch_points3d.datasets.change_detection.base_siamese_dataset import BaseSiameseDataset
from torch_points3d.datasets.change_detection.pair import Pair, MultiScalePair
from torch_points3d.metrics.change_detection_tracker import CDTracker
from torch_points3d.metrics.cls_cd_tracker import Cls_cd_tracker
from torch_points3d.datasets.change_detection.Urb3DSimulPairCylinder import Urb3DSimulCylinder, Urb3DSimul, to_ply


import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

IGNORE_LABEL: int = -1

URB3DCD_NUM_CLASSES = 7
URB3DCD_CLS_NUM_CLASSES = 5
viridis = cm.get_cmap('viridis', URB3DCD_CLS_NUM_CLASSES)

INV_OBJECT_LABEL = {
    0: "unchanged",
    1: "newlyBuilt",
    2: "deconstructed",
    3: "newVegetation",
    4: "vegetationRemoved",
}

# INV_OBJECT_LABEL = {i:"class " + str(i) for i in range(URB3DSIMUL_NUM_CLASSES)}
# V1
# OBJECT_COLOR = np.asarray(
#     [
#         [67, 1, 84],  # 'unchanged'
#         [0, 150, 128],  # 'newlyBuilt'
#         [255, 208, 0],  # 'deconstructed'
#
#     ]
# )

# V2
OBJECT_COLOR = np.asarray(
    [
        [67, 1, 84],  # 'unchanged'
        [0, 183, 255],  # 'newlyBuilt'
        [0, 12, 235],  # 'deconstructed'
        [0, 217, 33],  # 'newVegetation'
        [255, 230, 0],  # 'vegetationGrowUp'
        [255, 140, 0],  # 'vegetationRemoved'
        [255, 0, 0],  # 'mobileObjects'
    ]
)
OBJECT_LABEL = {name: i for i, name in INV_OBJECT_LABEL.items()}

class ImbalancedDatasetSampler(torch.utils.data.sampler.Sampler):
    """Samples elements randomly from a given list of indices for imbalanced dataset
    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback-like function which takes two arguments - dataset and index
    """

    def __init__(self, dataset, indices=None, num_samples=None, callback_get_label=None):

        # if indices is not provided,
        # all elements in the dataset will be considered
        self.indices = list(range(len(dataset))) \
            if indices is None else indices

        # define custom callback
        self.callback_get_label = callback_get_label

        # if num_samples is not provided,
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples
        # distribution of classes in the dataset
        label_to_count = {}
        for idx in self.indices:
            label = self._get_label(dataset, idx)
            if label in label_to_count:
                label_to_count[label] += 1
            else:
                label_to_count[label] = 1

        # weight for each sample
        weights = [1.0 / label_to_count[self._get_label(dataset, idx)]
                   for idx in self.indices]
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, dataset, idx):
        return dataset.__getitem__(idx).y.tolist()

    def __iter__(self):
        return (self.indices[i] for i in torch.multinomial(
            self.weights, self.num_samples, replacement=True))

    def __len__(self):
        return self.num_samples


class ImbalancedDatasetSampler_urb3DCD(ImbalancedDatasetSampler):
    def _get_label(self, dataset, idx):
        return dataset.labels[idx]

class Center(object):

    def __call__(self, data):
        data.pos[:, :3] = data.pos[:, :3] - data.pos[:, :3].mean(dim=-2, keepdim=True)
        data.pos_target[:, :3] = data.pos_target[:, :3] - data.pos_target[:, :3].mean(dim=-2, keepdim=True)
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)

class Urb3DCD_cls_cylinder_maker(Urb3DSimulCylinder):
    def __init__(self, *args, **kwargs):
        self.num_classes_Urb3DCD_seg = URB3DCD_NUM_CLASSES
        super(Urb3DSimulCylinder, self).__init__(*args, **kwargs)
        self.num_classes = URB3DCD_CLS_NUM_CLASSES

    def get_cls_class(self, labels, pos, pos0, id):
        pourc = get_pourc(labels, self.num_classes_Urb3DCD_seg)
        lab = 0
        max_ch_lab = max(pourc[1:])
        arg_max_chlab = np.argmax(pourc[1:])

        if max_ch_lab > 0.05:
            lab = arg_max_chlab + 1
        othercl = sum(pourc[1:]) - max_ch_lab
        if othercl > 0.03:
            lab = len(pourc)
        path = os.path.join(os.getcwd(), self.split, str(lab), id)
        if not os.path.isdir(path):
            os.makedirs(path)
        to_ply(pos, labels, os.path.join(path, id + "_PC1.ply"))
        to_ply(pos0, np.zeros(pos0.shape[0], dtype="int"), os.path.join(path, id + "_PC0.ply"))
        return lab

    def _prepare_centers(self):
        self._centres_for_sampling = []
        grid_sampling = GridSampling3D(size=self._radius * 2)
        self.grid_regular_centers = []
        for i in range(len(self.filesPC0)):
            pair = self._load_save(i)
            if self._sample_per_epoch > 0:
                dataPC0 = Data(pos=pair.pos)
                dataPC1 = Data(pos=pair.pos_target, y=pair.y)
                low_res = grid_sampling(dataPC1.clone())
                centres = torch.empty((low_res.pos.shape[0], 5), dtype=torch.float)
                for c in range(centres.shape[0]):
                    print(str(c) + '/' + str(centres.shape[0]), end='\r')

                    lab_cls = self.get_cyl_cls(dataPC1, low_res.pos[c, :], dataPC0, str(i) + "_" + str(c))
                    if lab_cls is not None:
                        centres[c, :3] = low_res.pos[c, :]
                        centres[c, 3] = i
                        centres[c, 4] = lab_cls
                    else:
                        centres[c,4] = -1
                centres = centres[centres[:,4]>=0]
                centres = centres[centres[:, 4]<7]
                self._centres_for_sampling.append(centres)
            else:
                # Get regular center on PC1, PC0 will be sampled using the same center
                dataPC1 = Data(pos=pair.pos_target, y=pair.y)
                grid_sample_centers = grid_sampling(dataPC1.clone())
                centres = torch.empty((grid_sample_centers.pos.shape[0], 4), dtype=torch.float)
                centres[:, :3] = grid_sample_centers.pos
                centres[:, 3] = i
                self.grid_regular_centers.append(centres)
        if self._sample_per_epoch > 0:
            self._centres_for_sampling = torch.cat(self._centres_for_sampling, 0)
            uni, uni_counts = np.unique(np.asarray(self._centres_for_sampling[:, -1]), return_counts=True)
            print(uni_counts)
            uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
            self._label_counts = uni_counts / np.sum(uni_counts)
            print(self._label_counts)
            self._labels = uni
            self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
        else:
            self.grid_regular_centers = torch.cat(self.grid_regular_centers, 0)

    def get_cyl_cls(self, data, center, data0, id):
        label_cls = None
        cylinder_sampler = CylinderSampling(self._radius, center, align_origin=False)
        cyl = cylinder_sampler(data)
        cyl0 = cylinder_sampler(data0)
        if cyl0.pos.shape[0] > 0 and cyl.pos.shape[0] > 0:
            label_cls = self.get_cls_class(cyl.y, cyl.pos, cyl0.pos, id)

        return label_cls


class Urb3DCD_cls_cylinder(Dataset):
    def __init__(self, filePaths="", split="train", DA=False, TTA = False, pre_transform=None,
                 nameInPly="params",sample_per_epoch=100, radius=2, fixed_nb_pts = -1 ):
        self.class_labels = OBJECT_LABEL
        self._ignore_label = IGNORE_LABEL
        self.nameInPly = nameInPly
        self.filePaths = filePaths
        self.split = split
        self.DA = DA
        self.TTA = TTA
        self.pre_transform = pre_transform
        self.num_classes = URB3DCD_CLS_NUM_CLASSES
        self.nb_elt_class = torch.zeros(self.num_classes)
        self.sample_per_epoch = sample_per_epoch
        self.get_path()
        uni, uni_counts = np.unique(np.asarray(self.labels), return_counts=True)
        print(uni_counts)
        uni_counts = np.sqrt(uni_counts.mean() / uni_counts)
        self._label_counts = uni_counts / np.sum(uni_counts)
        print(self._label_counts)
        self._labels = uni
        # self.weight_classes = torch.from_numpy(self._label_counts).type(torch.float)
        self.weight_classes = None
        self.fixed_nb_pts = fixed_nb_pts
        self.sample_pts = SamplePoints(self.fixed_nb_pts)


    def size(self):
        return self.__len__()

    def __len__(self):
        return len(self.filesPC1)

    def get_path(self):
        self.filesPC0 = []
        self.filesPC1 = []
        self.labels = []
        globPath = os.scandir(self.filePaths)
        for dir in globPath:
            if dir.is_dir():
                try:
                    lab = int(os.path.basename(dir))
                except:
                    continue
                curDir = os.scandir(dir)
                for dir2 in curDir:
                    if dir2.is_dir():
                        curDir2 = os.scandir(dir2)
                        for f in curDir2:
                            if "PC0.ply" in f.name:
                                self.filesPC0.append(f.path)
                            elif "PC1.ply" in f.name:
                                self.filesPC1.append(f.path)
                                self.labels.append(lab)

                curDir.close()
        globPath.close()

    def __getitem__(self, idx):
        pos0, pos1 = self._load(idx)
        area = os.path.dirname(self.filesPC0[idx]).split("/")[-1]
        batch = Pair(pos = pos0, pos_target = pos1, y = self.labels[idx], area=area)
        if self.fixed_nb_pts>0:
            batch = self.sample_pts(batch)
        if self.DA:
            batch.data_augment(paramGaussian=[0.005, 0.02], color_aug=False)
        if self.fixed_nb_pts < 0:
            batch.normalise()
        else:
            normaliser = Center()
            batch = normaliser(batch)
        return batch.contiguous()

    def _load(self, idx):
        pos0 = torch.tensor(read_from_ply(self.filesPC0[idx], self.nameInPly))
        pos1 = torch.tensor(read_from_ply(self.filesPC1[idx], self.nameInPly))
        return pos0, pos1






class Urb3DCDDataset_cls(BaseSiameseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.
        Parameters
        ----------
        dataset_opt: omegaconf.DictConfig
            Config dictionary that should contain
                - root,
                - split,
                - transform,
                - pre_transform
                - process_workers
        """
    INV_OBJECT_LABEL = INV_OBJECT_LABEL
    FORWARD_CLASS = "forward.urb3DSimulPairCyl.ForwardUrb3DSimulDataset"

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        self.radius = float(self.dataset_opt.radius)
        self.sample_per_epoch = int(self.dataset_opt.sample_per_epoch)
        self.DA = self.dataset_opt.DA
        self.preprocessed_dir = self.dataset_opt.preprocessed_dir

        self.train_dataset = Urb3DCD_cls_cylinder(
            filePaths=self.dataset_opt.dataTrainFile,
            split="train",
            radius=self.radius,
            sample_per_epoch=self.sample_per_epoch,
            DA=self.DA,
            pre_transform=self.pre_transform,
            fixed_nb_pts=self.dataset_opt.fixed_points if self.dataset_opt.fixed_points is not None else -1,
            nameInPly=self.dataset_opt.nameInPly,
        )
        self.val_dataset = Urb3DCD_cls_cylinder(
            filePaths=self.dataset_opt.dataValFile,
            split="val",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            fixed_nb_pts=self.dataset_opt.fixed_points if self.dataset_opt.fixed_points is not None else -1,
            nameInPly=self.dataset_opt.nameInPly,
        )
        self.test_dataset = Urb3DCD_cls_cylinder(
            filePaths=self.dataset_opt.dataTestFile,
            split="test",
            radius=self.radius,
            sample_per_epoch=-1,
            pre_transform=self.pre_transform,
            fixed_nb_pts= self.dataset_opt.fixed_points if self.dataset_opt.fixed_points is not None else -1,
            nameInPly=self.dataset_opt.nameInPly,
        )
        if self.dataset_opt.random_subsampleTrainset:
            self.train_sampler = ImbalancedDatasetSampler_urb3DCD(self.train_dataset, num_samples=self.sample_per_epoch)

    @property
    def train_data(self):
        if type(self.train_dataset) == list:
            return self.train_dataset[0]
        else:
            return self.train_dataset

    @property
    def val_data(self):
        if type(self.val_dataset) == list:
            return self.val_dataset[0]
        else:
            return self.val_dataset

    @property
    def test_data(self):
        if type(self.test_dataset) == list:
            return self.test_dataset[0]
        else:
            return self.test_dataset

    @staticmethod
    def to_ply(pos, label, file):
        """ Allows to save Urb3DCD predictions to disk using Urb3DCD color scheme
            Parameters
            ----------
            pos : torch.Tensor
                tensor that contains the positions of the points
            label : torch.Tensor
                predicted label
            file : string
                Save location
            """
        to_ply(pos, label, file)

    def get_tracker(self, wandb_log: bool, tensorboard_log: bool, full_pc=False, full_res=False):
        """Factory method for the tracker
            Arguments:
                wandb_log - Log using weight and biases
                tensorboard_log - Log using tensorboard
            Returns:
                [BaseTracker] -- tracker
            """
        return Cls_cd_tracker(self, wandb_log=wandb_log, use_tensorboard=tensorboard_log)


# UTILS

def get_pourc(labels, nb_classe):
    uni, uni_counts = np.unique(np.asarray(labels), return_counts=True)
    pourc = [0] * nb_classe
    som = uni_counts.sum()
    for cl in range(len(uni)):
        classe = uni[cl]
        pourc[classe] = uni_counts[cl] / som
    return pourc

def read_from_ply(filename, nameInPly="vertex"):
    """read XYZ for each vertex."""
    assert os.path.isfile(filename)
    with open(filename, "rb") as f:
        plydata = PlyData.read(f)
        num_verts = plydata[nameInPly].count
        pos = np.zeros(shape=[num_verts, 3], dtype=np.float32)
        pos[:, 0] = plydata[nameInPly].data["x"]
        pos[:, 1] = plydata[nameInPly].data["y"]
        pos[:, 2] = plydata[nameInPly].data["z"]
    return pos


class SamplePoints(object):
    def __init__(self, num: int) -> None:
        super().__init__()
        self.num = num

    def __call__(self, data):
        x1, x2 = data.pos, data.pos_target
        assert x1.size(1) >= 3 and x2.size(1) >= 3

        if self.num < x1.size(0):
            idx = torch.randperm(x1.size(0))[:self.num]
            x1 = x1[idx]
        if self.num < x2.size(0):
            idx = torch.randperm(x2.size(0))[:self.num]
            x2 = x2[idx]

        data.pos, data.pos_target = x1, x2
        return data