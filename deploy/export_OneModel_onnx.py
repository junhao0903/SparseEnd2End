# Copyright (c) 2024 SparseEnd2End. All rights reserved @author: Thomas Von Wu.
import os
import sys
import time
import copy
import logging
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import onnx
from onnxsim import simplify

import torch
from torch import nn

from modules.sparse4d_detector import *
from modules.head.sparse4d_blocks.instance_bank import topk
from modules.ops import deformable_aggregation_function as DAF

from tool.utils.config import read_cfg
from typing import Optional, Dict, Any

from tool.utils.logger import set_logger


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy SparseEND2END Head!")
    parser.add_argument(
        "--cfg",
        type=str,
        default="dataset/config/sparse4d_temporal_r50_1x1_bs1_256x704_mini.py",
        help="deploy config file path",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="ckpt/sparse4dv3_r50.pth",
        help="deploy ckpt path",
    )
    parser.add_argument(
        "--use_deformable_func",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="是否使用cuda算子，非nvidia芯片均不采用cuda算子，走grid_sample支路",
    )
    parser.add_argument(
        "--log",
        type=str,
        default="deploy/onnx/export_head_onnx.log",
    )
    parser.add_argument(
        "--save_onnx1",
        type=str,
        default="deploy/onnx/sparse4dv3_1st.onnx",
    )
    parser.add_argument(
        "--save_onnx2",
        type=str,
        default="deploy/onnx/sparse4dv3_2nd.onnx",
    )
    parser.add_argument(
        "--o2", action="store_true", help="only export sparse4dv32nd onnx."
    )
    args = parser.parse_args()
    return args


class Sparse4Dv31st(nn.Module):
    def __init__(self, model, use_deformable_func=False):
        super(Sparse4Dv31st, self).__init__()
        self.model = model
        self.use_deformable_func = use_deformable_func

        # ======================
        # 只存数值，不存张量！
        # ======================
        self.img_w = 704
        self.img_h = 256
        self.num_cams = 6

    @staticmethod
    def head_forward(
        self,
        feature_maps,
        instance_feature,
        anchor,
        time_interval,
        image_wh,
        lidar2img,
        use_deformable_func = False
    ):

        # Instance bank get inputs
        temp_instance_feature = None
        temp_anchor_embed = None

        # DFA inputs
        metas = {
            "image_wh": image_wh,
            "lidar2img": lidar2img,
        }

        anchor_embed = self.anchor_encoder(anchor)

        prediction = []
        for i, op in enumerate(self.operation_order):
            print("i: ", i, "\top: ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(
                    instance_feature, anchor_embed, metas
                )
                if use_deformable_func:
                    points_2d = (
                        self.layers[i]
                        .project_points(
                            key_points,
                            metas["lidar2img"],  # lidar2img
                            metas.get("image_wh"),
                        )
                        .permute(0, 2, 3, 1, 4)
                        .reshape(bs, num_anchor, self.layers[i].num_pts, self.layers[i].num_cams, 2)
                    )
                    weights = (
                        weights.permute(0, 1, 4, 2, 3, 5)
                        .contiguous()
                        .reshape(
                            bs,
                            num_anchor,
                            self.layers[i].num_pts,
                            self.layers[i].num_cams,
                            self.layers[i].num_levels,
                            self.layers[i].num_groups,
                        )
                    )
                    features = DAF(*feature_maps, points_2d, weights).reshape(
                        bs, num_anchor, self.layers[i].embed_dims
                    )

                else:
                    features = self.layers[i].feature_sampling(
                        feature_maps,
                        key_points,
                        metas["lidar2img"],
                        metas.get("image_wh"),
                    )
                    features = self.layers[i].multi_view_level_fusion(features, weights)
                    features = features.sum(dim=2)

                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)
                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
        return (
            instance_feature,
            anchor,
            cls,
            qt,
        )

    def forward(
        self,
        img,
        instance_feature,
        anchor,
        time_interval,
        lidar2img,
    ):
        bs = img.shape[0]
        image_wh = torch.tensor([[self.img_w, self.img_h]],
                                device=img.device,
                                dtype=torch.float32)
        image_wh = image_wh.repeat(bs, self.num_cams, 1)

        feature_maps = self.model.extract_feat(img)
        head = self.model.head
        return self.head_forward(
            head,
            feature_maps,
            instance_feature,
            anchor,
            time_interval,
            image_wh,
            lidar2img,
            self.use_deformable_func
        )


class Sparse4Dv32nd(nn.Module):
    def __init__(self, model, use_deformable_func=False):
        super(Sparse4Dv32nd, self).__init__()
        self.model = model
        self.use_deformable_func = use_deformable_func
        self.img_w = 704
        self.img_h = 256
        self.num_cams = 6

    @staticmethod
    def head_forward(
        self,
        feature_maps,
        instance_feature,
        anchor,
        time_interval,
        temp_instance_feature,
        temp_anchor,
        mask,
        track_id,
        image_wh,
        lidar2img,
        use_deformable_func=False
    ):
        mask = mask.bool()  # TensorRT binding type for bool input is NoneType.
        anchor_embed = self.anchor_encoder(anchor)
        temp_anchor_embed = self.anchor_encoder(temp_anchor)

        # DAF inputs
        metas = {
            "lidar2img": lidar2img,
            "image_wh": image_wh,
        }

        prediction = []
        for i, op in enumerate(self.operation_order):
            print("op:  ", op)
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    temp_instance_feature,
                    temp_instance_feature,
                    query_pos=anchor_embed,
                    key_pos=temp_anchor_embed,
                )
            elif op == "gnn":
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    value=instance_feature,
                    query_pos=anchor_embed,
                )
            elif op == "norm" or op == "ffn":
                instance_feature = self.layers[i](instance_feature)
            elif op == "deformable":
                bs, num_anchor = instance_feature.shape[:2]
                key_points = self.layers[i].kps_generator(anchor, instance_feature)
                weights = self.layers[i]._get_weights(
                    instance_feature, anchor_embed, metas
                )
                if use_deformable_func:
                    points_2d = (
                        self.layers[i]
                        .project_points(
                            key_points,
                            metas["lidar2img"],  # lidar2img
                            metas.get("image_wh"),
                        )
                        .permute(0, 2, 3, 1, 4)
                        .reshape(bs, num_anchor, self.layers[i].num_pts, self.layers[i].num_cams, 2)
                    )
                    weights = (
                        weights.permute(0, 1, 4, 2, 3, 5)
                        .contiguous()
                        .reshape(
                            bs,
                            num_anchor,
                            self.layers[i].num_pts,
                            self.layers[i].num_cams,
                            self.layers[i].num_levels,
                            self.layers[i].num_groups,
                        )
                    )

                    features = DAF(*feature_maps, points_2d, weights).reshape(
                        bs, num_anchor, self.layers[i].embed_dims
                    )
                else:
                    features = self.layers[i].feature_sampling(
                        feature_maps,
                        key_points,
                        metas["lidar2img"],
                        metas.get("image_wh"),
                    )
                    features = self.layers[i].multi_view_level_fusion(features, weights)
                    features = features.sum(dim=2)

                output = self.layers[i].output_proj(features)
                assert self.layers[i].residual_mode == "cat"
                instance_feature = torch.cat([output, instance_feature], dim=-1)
            elif op == "refine":
                anchor, cls, qt = self.layers[i](
                    instance_feature,
                    anchor,
                    anchor_embed,
                    time_interval=time_interval,
                    return_cls=(
                        len(prediction) == self.num_single_frame_decoder - 1
                        or i == len(self.operation_order) - 1
                    ),
                )
                prediction.append(anchor)

                # update in head refine
                if len(prediction) == self.num_single_frame_decoder:
                    N = (
                        self.instance_bank.num_anchor
                        - self.instance_bank.num_temp_instances
                    )
                    cls = cls.max(dim=-1).values
                    _, (selected_feature, selected_anchor), _ = topk(
                        cls, N, instance_feature, anchor
                    )
                    selected_feature = torch.cat(
                        [temp_instance_feature, selected_feature], dim=1
                    )
                    selected_anchor = torch.cat([temp_anchor, selected_anchor], dim=1)
                    instance_feature = torch.where(
                        mask[:, None, None], selected_feature, instance_feature
                    )
                    anchor = torch.where(mask[:, None, None], selected_anchor, anchor)
                    track_id = torch.where(
                        mask[:, None],
                        track_id,
                        track_id.new_tensor(-1),
                    )

                if i != len(self.operation_order) - 1:
                    anchor_embed = self.anchor_encoder(anchor)
                if len(prediction) > self.num_single_frame_decoder:
                    temp_anchor_embed = anchor_embed[
                        :, : self.instance_bank.num_temp_instances
                    ]
        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id
        )

    def forward(
        self,
        img,
        instance_feature,
        anchor,
        time_interval,
        temp_instance_feature,
        temp_anchor,
        mask,
        track_id,
        lidar2img,
    ):
        bs = img.shape[0]
        image_wh = torch.tensor([[self.img_w, self.img_h]],
                                device=img.device,
                                dtype=torch.float32)
        image_wh = image_wh.repeat(bs, self.num_cams, 1)

        feature_maps = self.model.extract_feat(img)
        head = self.model.head
        (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id
        ) = self.head_forward(
            head,
            feature_maps,
            instance_feature,
            anchor,
            time_interval,
            temp_instance_feature,
            temp_anchor,
            mask,
            track_id,
            image_wh,
            lidar2img,
            self.use_deformable_func
        )
        return (
            instance_feature,
            anchor,
            cls,
            qt,
            track_id
        )


def dummpy_input(
    model,
    bs: int,
    nums_cam: int,
    input_h: int,
    input_w: int,
    nums_query=900,
    nums_topk=600,
    embed_dims=256,
    anchor_dims=11,
    first_frame=True,
    logger=None,
):
    h_4x, w_4x = input_h // 4, input_w // 4
    h_8x, w_8x = input_h // 8, input_w // 8
    h_16x, w_16x = input_h // 16, input_w // 16
    h_32x, w_32x = input_h // 32, input_w // 32
    feature_size = nums_cam * (
        h_4x * w_4x + h_8x * w_8x + h_16x * w_16x + h_32x * w_32x
    )
    dummy_imgs = torch.randn(bs, nums_cam, 3, input_h, input_w).float().cuda()

    instance_feature = model.head.instance_bank.instance_feature  # (900, 256)
    dummy_instance_feature = (
        instance_feature[None].repeat((bs, 1, 1)).cuda()
    )  # (bs, 900, 256)

    anchor = model.head.instance_bank.anchor  # (900, 11)
    dummy_anchor = anchor[None].repeat((bs, 1, 1)).cuda()  # (bs, 900, 11)

    dummy_time_interval = torch.tensor(
        [model.head.instance_bank.default_time_interval] * bs
    ).cuda()

    dummy_temp_instance_feature = (
        torch.zeros((bs, nums_topk, embed_dims)).float().cuda()
    )
    dummy_temp_anchor = torch.zeros((bs, nums_topk, anchor_dims)).float().cuda()
    dummy_mask = torch.randint(0, 2, size=(bs,)).int().cuda()
    dummy_track_id = -1 * torch.ones((bs, nums_query)).int().cuda()


    dummy_lidar2img = torch.randn(bs, nums_cam, 4, 4).to(dummy_imgs)

    logger.debug(f"Dummy input : hape&Type&Device Msg >>>>>>")
    roi_x = [
        "dummy_imgs",
        "dummy_instance_feature",
        "dummy_anchor",
        "dummy_time_interval",
        "dummy_lidar2img",
    ]
    for x in roi_x:
        logger.debug(
            f"{x}\t:\tshape={eval(x).shape},\tdtype={eval(x).dtype},\tdevice={eval(x).device}"
        )

    if first_frame:
        logger.debug(f"Frame > 1: Extra dummy input is needed >>>>>>>")
        roi_y = [
            "dummy_temp_instance_feature",
            "dummy_temp_anchor",
            "dummy_mask",
            "dummy_track_id",
        ]
        for y in roi_y:
            logger.debug(
                f"{y}\t:\tshape={eval(y).shape},\tdtype={eval(y).dtype},\tdevice={eval(y).device}"
            )

    return (
        dummy_imgs,
        dummy_instance_feature,
        dummy_anchor,
        dummy_time_interval,
        dummy_temp_instance_feature,
        dummy_temp_anchor,
        dummy_mask,
        dummy_track_id,
        dummy_lidar2img,
    )


def build_module(cfg, default_args: Optional[Dict] = None) -> Any:
    cfg2 = cfg.copy()
    if default_args is not None:
        for name, value in default_args.items():
            cfg2.setdefault(name, value)
    type = cfg2.pop("type")
    return eval(type)(**cfg2)


def override_use_deformable_func(cfg, use_deformable_func):
    if isinstance(cfg, dict):
        for key, value in cfg.items():
            if key == "use_deformable_func":
                cfg[key] = use_deformable_func
            else:
                override_use_deformable_func(value, use_deformable_func)
    elif isinstance(cfg, list):
        for item in cfg:
            override_use_deformable_func(item, use_deformable_func)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(os.path.dirname(args.save_onnx1), exist_ok=True)
    logger, console_handler, file_handler = set_logger(args.log, True)
    logger.setLevel(logging.DEBUG)
    console_handler.setLevel(logging.DEBUG)
    file_handler.setLevel(logging.DEBUG)
    cfg = read_cfg(args.cfg)
    use_deformable_func = args.use_deformable_func
    if use_deformable_func != cfg["model"].get("use_deformable_func", True):
        override_use_deformable_func(cfg["model"], use_deformable_func)

    if use_deformable_func:
        print("cuda算子支路")
    else:
        print("grid sample支路")
        args.save_onnx1 = args.save_onnx1.replace(".onnx", "_gridsample.onnx")
        args.save_onnx2 = args.save_onnx2.replace(".onnx", "_gridsample.onnx")

    model = build_module(cfg["model"])
    checkpoint = args.ckpt
    _ = model.load_state_dict(torch.load(checkpoint)["state_dict"], strict=False)
    model.cuda().eval()

    BS = 1
    NUMS_CAM = 6
    INPUT_H = 256
    INPUT_W = 704
    first_frame = True
    (
        dummy_imgs,
        dummy_instance_feature,
        dummy_anchor,
        dummy_time_interval,
        dummy_temp_instance_feature,
        dummy_temp_anchor,
        dummy_mask,
        dummy_track_id,
        dummy_lidar2img,
    ) = dummpy_input(
        model, BS, NUMS_CAM, INPUT_H, INPUT_W, first_frame=first_frame, logger=logger
    )

    if not args.o2:
        first_frame_head = Sparse4Dv31st(copy.deepcopy(model), use_deformable_func)
        logger.info("Export Sparse4Dv31st Onnx >>>>>>>>>>>>>>>>")
        time.sleep(2)
        with torch.no_grad():
            torch.onnx.export(
                first_frame_head,
                (
                    dummy_imgs,
                    dummy_instance_feature,
                    dummy_anchor,
                    dummy_time_interval,
                    dummy_lidar2img,
                ),
                args.save_onnx1,
                input_names=[
                    "img",
                    "instance_feature",
                    "anchor",
                    "time_interval",
                    "lidar2img",
                ],
                output_names=[
                    "pred_instance_feature",
                    "pred_anchor",
                    "pred_class_score",
                    "pred_quality_score"
                ],
                opset_version=16,
                do_constant_folding=True,
                verbose=False,
                training=torch.onnx.TrainingMode.EVAL,  # ✅ 禁用训练
                keep_initializers_as_inputs=False,  # ✅ 减少显存
            )

            onnx_orig = onnx.load(args.save_onnx1)
            onnx_simp, check = simplify(onnx_orig)
            assert check, "Simplified ONNX model could not be validated"
            onnx.save(onnx_simp, args.save_onnx1)
            logger.info(
                f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx1}" 🤗.'
            )

    head = Sparse4Dv32nd(copy.deepcopy(model), use_deformable_func)
    logger.info("Export Sparse4Dv32nd Onnx >>>>>>>>>>>>>>>>")
    time.sleep(2)
    with torch.no_grad():
        torch.onnx.export(
            head,
            (
                dummy_imgs,
                dummy_instance_feature,
                dummy_anchor,
                dummy_time_interval,
                dummy_temp_instance_feature,
                dummy_temp_anchor,
                dummy_mask,
                dummy_track_id,
                dummy_lidar2img,
            ),
            args.save_onnx2,
            input_names=[
                "img",
                "instance_feature",
                "anchor",
                "time_interval",
                "temp_instance_feature",
                "temp_anchor",
                "mask",
                "track_id",
                "lidar2img",
            ],
            output_names=[
                "pred_instance_feature",
                "pred_anchor",
                "pred_class_score",
                "pred_quality_score",
                "pred_track_id",
            ],
            opset_version=16,
            do_constant_folding=True,
            verbose=False,
        )

        onnx_orig = onnx.load(args.save_onnx2)
        onnx_simp, check = simplify(onnx_orig)
        assert check, "Simplified ONNX model could not be validated!"
        onnx.save(onnx_simp, args.save_onnx2)
        logger.info(f'🚀 Export onnx completed. ONNX saved in "{args.save_onnx2}" 🤗.')
