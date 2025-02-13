from typing import *

import json
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.data import transforms as T, build_detection_train_loader, detection_utils as utils
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch


class ImageCodeDataset(Dataset):

    def __init__(self,
                 image_path: str,
                 code_path: str,
                 split,
                 *,
                #  transform = None,
                 label_trans_path = None):
        super().__init__()
        # self.transform = T.Compose([
        #     T.ToDtype(torch.float, scale=True),
        #     T.Normalize(mean=mean, std=std),
        #     T.ToPureTensor()
        # ])
        with open(label_trans_path, "r") as input:
            self.label_trans = json.load(input)

        self.hi = h5py.File(image_path, "r")
        self.images = self.hi["images"]
        self.labels = self.hi["labels"]   
        self.rects = self.hi["rects"]

        self.hc = h5py.File(code_path, "r")
        self.idx = self.hc["idx"]
        self.pid = self.hc["pid"]
        self.piv = self.hc["piv"]

        self.split = np.unique(self.idx[split])
        # print(self.split)
        assert len(self.split) < len(self.images), (len(self.split), len(self.images))

    def __len__(self) -> int:
        if self.split is not None:
            return len(self.split)
        return len(self.images) 
    
    def __idx(self, i: int) -> int:
        if self.split is not None:
            return self.split[i]
        return i
    
    def __getitem__(self, index: int) -> Dict:
        img_idx = self.__idx(index)
        image = torch.from_numpy(self.images[img_idx])
        # if self.transform is not None:
        #     image = self.transform(image)

        annotations = []

        indices = np.where(self.idx == img_idx)[0]
        for i in indices:
            pid = self.pid[i]
            piv = self.piv[i]
            if pid != -1:
                loc = np.where(np.logical_and(
                    self.labels[:, 0] == img_idx,
                    self.labels[:, 1] == pid
                ))
                assert len(loc[0]) == 1
                rect = self.rects[loc[0]][0]
            else:
                rect = (0, 0, image.size(1) - 1, image.size(2) - 1)

            category_id = self.label_trans[piv][-1]
            assert category_id in [8, 40]

            annotations.append({
                "bbox": rect,
                "bbox_mode": 0,  # BoxMode.XYXY_ABS
                "category_id": 0 if category_id == 8 else 1,
            })

        return {
            "image": image,
            "height": 256,
            "width": 256,
            "image_id": img_idx,
            "annotations": annotations
        }


class DatasetMapper:

    def __init__(self, cfg, is_train):
        self.augmentations = utils.build_augmentation(cfg, is_train)

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape
        )

        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        # dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        aug_input = T.AugInput(dataset_dict["image"])
        transforms = self.augmentations(aug_input)
        image = aug_input.image

        image_shape = image.shape[:2]  # h, w

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict


class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains a number pre-defined logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can use the cleaner
    "SimpleTrainer", or write your own training loop.
    """

    @classmethod
    def build_train_loader(cls, cfg):
        split = np.load(cfg.DATASETS.SPLIT_PATH)["train"]
        dataset = ImageCodeDataset(cfg.DATASETS.IMAGE_PATH,
                                   cfg.DATASETS.CODE_PATH,
                                   split,
                                   label_trans_path=cfg.DATASETS.LABEL_TRANS_PATH)
        dataset_mapper = DatasetMapper(cfg, is_train=True)
        return build_detection_train_loader(cfg, dataset=dataset, mapper=dataset_mapper)

    # @classmethod
    # def build_lr_scheduler(cls, cfg, optimizer):
    #     """
    #     It now calls :func:`detectron2.solver.build_lr_scheduler`.
    #     Overwrite it if you'd like a different scheduler.
    #     """
    #     return build_lr_scheduler(cfg, optimizer)


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    
    #
    cfg.DATASETS.IMAGE_PATH = ""
    cfg.DATASETS.CODE_PATH = ""
    cfg.DATASETS.SPLIT_PATH = ""
    cfg.DATASETS.LABEL_TRANS_PATH = ""

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover