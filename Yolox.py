import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, List

import configparser
from typing import Dict, Any


def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

def parse_yolox_cfg(cfg_path: str) -> Dict[str, Any]:
    """Parse YOLOX configuration file."""
    config = configparser.ConfigParser()
    config.read(cfg_path)

    cfg_dict = {
        # Network configuration
        'width': config.getint('net', 'width', fallback=640),
        'height': config.getint('net', 'height', fallback=640),
        'embedding_dim': config.getint('net', 'embedding_dim', fallback=512),
        'channels': config.getint('net', 'channels', fallback=3),
        'depth': config.getfloat('net', 'depth', fallback=1.0),
        'width_mul': config.getfloat('net', 'width_mul', fallback=1.0),

        # Model configuration
        'num_classes': config.getint('model', 'num_classes', fallback=1),
        'strides': [int(x) for x in config.get('model', 'strides', fallback='8,16,32').split(',')],
        'use_l1': config.getboolean('model', 'use_l1', fallback=False),
        'use_reid': config.getboolean('model', 'use_reid', fallback=True),
        'reid_dim': config.getint('model', 'reid_dim', fallback=512),

        # Training configuration
        'batch_size': config.getint('train', 'batch', fallback=16),
        'subdivisions': config.getint('train', 'subdivisions', fallback=1),
        'momentum': config.getfloat('train', 'momentum', fallback=0.9),
        'weight_decay': config.getfloat('train', 'weight_decay', fallback=0.0005),
        'ignore_thresh': config.getfloat('train', 'ignore_thresh', fallback=0.7),
        'truth_thresh': config.getfloat('train', 'truth_thresh', fallback=1.0),
        'random': config.getint('train', 'random', fallback=1),
    }

    return cfg_dict

class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
            self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DWConv(nn.Module):
    """Depthwise Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


class Focus(nn.Module):
    """Focus width and height information into channel space."""

    def __init__(self, in_channels, out_channels, ksize=1, stride=1, act="silu"):
        super().__init__()
        self.conv = BaseConv(in_channels * 4, out_channels, ksize, stride, act=act)

    def forward(self, x):
        patch_top_left = x[..., ::2, ::2]
        patch_top_right = x[..., ::2, 1::2]
        patch_bot_left = x[..., 1::2, ::2]
        patch_bot_right = x[..., 1::2, 1::2]
        x = torch.cat(
            (patch_top_left, patch_bot_left, patch_top_right, patch_bot_right,), dim=1,
        )
        return self.conv(x)


class CSPLayer(nn.Module):
    """CSP Bottleneck with 3 convolutions"""

    def __init__(
            self, in_channels, out_channels, n=1, shortcut=True, expansion=0.5, act="silu"
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv3 = BaseConv(2 * hidden_channels, out_channels, 1, stride=1, act=act)
        module_list = [
            Bottleneck(hidden_channels, hidden_channels, shortcut, 1.0, act=act)
            for _ in range(n)
        ]
        self.m = nn.Sequential(*module_list)

    def forward(self, x):
        x_1 = self.conv1(x)
        x_2 = self.conv2(x)
        x_1 = self.m(x_1)
        x = torch.cat((x_1, x_2), dim=1)
        return self.conv3(x)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self,
        in_channels,
        out_channels,
        shortcut=True,
        expansion=0.5,
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        hidden_channels = int(out_channels * expansion)
        Conv = DWConv if depthwise else BaseConv
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.conv2 = Conv(hidden_channels, out_channels, 3, stride=1, act=act)
        self.use_add = shortcut and in_channels == out_channels

    def forward(self, x):
        y = self.conv2(self.conv1(x))
        if self.use_add:
            y = y + x
        return y


class SPPBottleneck(nn.Module):
    """Spatial pyramid pooling layer used in YOLOv3-SPP"""

    def __init__(self, in_channels, out_channels, kernel_sizes=(5, 9, 13), act="silu"):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv1 = BaseConv(in_channels, hidden_channels, 1, stride=1, act=act)
        self.m = nn.ModuleList([
            nn.MaxPool2d(kernel_size=ks, stride=1, padding=ks // 2)
            for ks in kernel_sizes
        ])
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv2 = BaseConv(conv2_channels, out_channels, 1, stride=1, act=act)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.cat([x] + [m(x) for m in self.m], dim=1)
        x = self.conv2(x)
        return x


class CSPDarknet(nn.Module):
    def __init__(
            self, dep_mul, wid_mul, out_features=("dark3", "dark4", "dark5"),
            act="silu", depthwise=False, nID=0
    ):
        super().__init__()
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2, base_channels * 2,
                n=base_depth, act=act
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4, base_channels * 4,
                n=base_depth * 3, act=act
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8, base_channels * 8,
                n=base_depth * 3, act=act
            ),
        )

        # dark5
        self.dark5 = nn.Sequential(
            Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
            SPPBottleneck(base_channels * 16, base_channels * 16, act=act),
            CSPLayer(
                base_channels * 16, base_channels * 16,
                n=base_depth, shortcut=False, act=act
            ),
        )

        # ReID support
        self.nID = nID
        if nID > 0:
            self.emb_dim = base_channels * 16  # Using the last layer's channels
            self.classifier = nn.Linear(self.emb_dim, nID)
            self.emb_scale = math.sqrt(2) * math.log(nID - 1) if nID > 1 else 1
            self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        else:
            self.classifier = None

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}


import torch
import torch.nn as nn
from typing import Dict, Any


class YOLOX(nn.Module):
    """
    YOLOX object detection model with ReID support
    """

    def __init__(self, cfg_dict: Dict[str, Any], nID: int = 0):
        super().__init__()
        self.cfg = cfg_dict
        self.nID = nID

        # Initialize backbone
        self.backbone = CSPDarknet(
            dep_mul=self.cfg['depth'],
            wid_mul=self.cfg['width_mul'],
            out_features=("dark3", "dark4", "dark5"),
            act="silu",
            nID=nID if self.cfg['use_reid'] else 0
        )

        # Initialize detection head
        self.head = YOLOXHead(
            num_classes=self.cfg['num_classes'],
            strides=self.cfg['strides'],
            in_channels=[256, 512, 1024],
            act="silu",
            embedding_dim=self.cfg['reid_dim'] if self.cfg['use_reid'] else 0
        )

        # Initialize ReID classifier if needed
        if self.cfg['use_reid'] and nID > 0:
            self.reid_head = nn.Linear(self.cfg['reid_dim'], nID)
            self.emb_scale = math.sqrt(2) * math.log(nID - 1) if nID > 1 else 1
        else:
            self.reid_head = None

        # Training parameters
        self.use_l1 = self.cfg['use_l1']

    def forward(self, x, targets=None):
        # Get features from backbone
        fpn_outs = self.backbone(x)

        if self.training:
            assert targets is not None
            outputs = self.head(fpn_outs, targets)

            # Add ReID loss if needed
            if self.reid_head is not None:
                reid_features = outputs['reid_features']
                if reid_features is not None:
                    reid_features = self.emb_scale * F.normalize(reid_features, dim=1)
                    outputs['reid_logits'] = self.reid_head(reid_features)

            return outputs
        else:
            outputs = self.head(fpn_outs)
            return outputs

    def load_config(self, cfg_path: str):
        """Load configuration from file"""
        new_cfg = parse_yolox_cfg(cfg_path)
        self.cfg.update(new_cfg)

    def get_model_info(self):
        """Get model configuration info"""
        info = {
            'input_size': (self.cfg['height'], self.cfg['width']),
            'num_classes': self.cfg['num_classes'],
            'use_reid': self.cfg['use_reid'],
            'reid_dim': self.cfg['reid_dim'],
            'depth': self.cfg['depth'],
            'width_mul': self.cfg['width_mul'],
            'strides': self.cfg['strides']
        }
        return info


class IOUloss(nn.Module):
    def __init__(self, reduction="none", loss_type="iou"):
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl = torch.max(pred[:, :2], target[:, :2])
        br = torch.min(pred[:, 2:], target[:, 2:])
        area_p = torch.prod(pred[:, 2:] - pred[:, :2], 1)
        area_g = torch.prod(target[:, 2:] - target[:, :2], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(pred[:, :2], target[:, :2])
            c_br = torch.max(pred[:, 2:], target[:, 2:])
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class YOLOXHead(nn.Module):
    def __init__(
            self,
            num_classes,
            strides=[8, 16, 32],
            in_channels=[256, 512, 1024],
            act="silu",
            depthwise=False,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.strides = strides
        self.in_channels = in_channels

        # Define the detection head
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.obj_preds = nn.ModuleList()
        self.stems = nn.ModuleList()

        for in_channel in in_channels:
            self.stems.append(BaseConv(in_channel, in_channel // 2, ksize=1, stride=1, act=act))
            self.cls_convs.append(nn.Sequential(*[
                BaseConv(in_channel // 2, in_channel // 2, ksize=3, stride=1, act=act),
                BaseConv(in_channel // 2, in_channel // 2, ksize=3, stride=1, act=act),
            ]))
            self.reg_convs.append(nn.Sequential(*[
                BaseConv(in_channel // 2, in_channel // 2, ksize=3, stride=1, act=act),
                BaseConv(in_channel // 2, in_channel // 2, ksize=3, stride=1, act=act),
            ]))
            self.cls_preds.append(
                nn.Conv2d(in_channel // 2, self.num_classes, kernel_size=1, stride=1, padding=0)
            )
            self.reg_preds.append(
                nn.Conv2d(in_channel // 2, 4, kernel_size=1, stride=1, padding=0)
            )
            self.obj_preds.append(
                nn.Conv2d(in_channel // 2, 1, kernel_size=1, stride=1, padding=0)
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")

    def forward(self, xin, labels=None, imgs=None):
        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        # Process each detection head
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
                zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = torch.cat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].type()
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])
                expanded_strides.append(
                    torch.zeros(1, grid.shape[1])
                    .fill_(stride_this_level)
                    .type_as(xin[0])
                )
                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.view(
                        batch_size, 1, 4, hsize, wsize
                    )
                    reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(
                        batch_size, -1, 4
                    )
                    origin_preds.append(reg_output.clone())

            else:
                output = torch.cat(
                    [reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1
                )

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = torch.cat(
                [x.flatten(start_dim=2) for x in outputs], dim=2
            ).permute(0, 2, 1)
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].type())
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]
        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, n_ch, hsize, wsize)
        output = output.permute(0, 2, 3, 1).reshape(
            batch_size, hsize * wsize, -1
        )
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4:5]  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            num_gts += num_gt
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                    )
                except RuntimeError:
                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        obj_preds,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.to(torch.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.type(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = max(num_fg, 1)
        loss_iou = (
                       self.iou_loss(bbox_preds.view(-1, 4)[fg_masks], reg_targets)
                   ).sum() / num_fg
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.view(-1, 1), obj_targets)
                   ).sum() / num_fg
        loss_cls = (
                       self.bcewithlog_loss(
                           cls_preds.view(-1, self.num_classes)[fg_masks], cls_targets
                       )
                   ).sum() / num_fg
        if self.use_l1:
            loss_l1 = (
                          self.l1_loss(origin_preds.view(-1, 4)[fg_masks], l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )
