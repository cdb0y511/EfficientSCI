import os
import os.path as osp

from torch.onnx.symbolic_opset8 import empty
from torch.utils.data import Dataset
import h5py
import einops
import numpy as np
import scipy.io as scio
from .builder import DATASETS


@DATASETS.register_module
class MatlabNerfRGBData(Dataset):
    def __init__(self, data_root, *args, **kwargs):
        self.data_root = data_root
        self.data_name_list = os.listdir(data_root)

        if kwargs["rot_flip_flag"]:
            self.rot_flip_flag = True
        else:
            self.rot_flip_flag = False
        if "transpose" in kwargs.keys():
            self.transpose = True
        else:
            self.transpose = False
        if "partition" in kwargs.keys():
            self.partition = kwargs["partition"]
        else:
            self.partition = None
        self.mask_lists = kwargs["mask_list"]

    def __getitem__(self, index):
        try:
            data = scio.loadmat(osp.join(self.data_root, self.data_name_list[index]))
            meas = data["meas"]

        except:
            data = h5py.File(osp.join(self.data_root, self.data_name_list[index]))
            meas = data["meas"]
        meas = einops.rearrange(meas, "h w c->c h w ")

        if "orig" in data.keys():
            pic_gt = einops.rearrange(data["orig"], "h w c cr->c cr h w")
            return meas, pic_gt
        else:
            return meas

    def __len__(self, ):
        return len(self.data_name_list)
