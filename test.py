import argparse
import json
import time
from pathlib import Path

import torchvision
from sklearn import metrics
from scipy import interpolate
import torch.nn.functional as F
from Yolox import YOLOX, parse_yolox_cfg  # 改为导入YOLOX模型
from utils.utils import *
from torchvision.transforms import transforms as T
from utils.datasets import LoadImagesAndLabels, JointDataset, collate_fn


def test(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
):
    # Configure run
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    nC = 1  # number of classes
    test_path = data_cfg_dict['test']
    dataset_root = data_cfg_dict['root']

    # 解析YOLOX配置
    cfg_dict = parse_yolox_cfg(cfg)
    img_size = [cfg_dict['width'], cfg_dict['height']]

    # Initialize YOLOX model
    model = YOLOX(cfg_dict)

    # Load weights
    if weights.endswith('.pt'):  # pytorch format
        checkpoint = torch.load(weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, test_path, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=collate_fn
    )

    mean_mAP, mean_R, mean_P, seen = 0.0, 0.0, 0.0, 0
    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    outputs, mAPs, mR, mP = [], [], [], []
    AP_accum, AP_accum_count = np.zeros(nC), np.zeros(nC)

    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        # YOLOX推理
        outputs = model(imgs.cuda())
        # YOLOX的输出处理
        outputs = postprocess(
            outputs,
            num_classes=cfg_dict['num_classes'],
            conf_thre=conf_thres,
            nms_thre=nms_thres
        )

        # 处理每个图像的结果
        targets = [targets[i][:int(l)] for i, l in enumerate(targets_len)]
        for si, (labels, detections) in enumerate(zip(targets, outputs)):
            seen += 1

            if detections is None:
                if labels.size(0) != 0:
                    mAPs.append(0), mR.append(0), mP.append(0)
                continue

            # 处理YOLOX的检测结果
            detections = detections.cpu().numpy()
            if len(detections):
                detections = detections[np.argsort(-detections[:, 4])]

            # 如果没有标签，添加检测作为不正确的
            correct = []
            if labels.size(0) == 0:
                mAPs.append(0), mR.append(0), mP.append(0)
                continue
            else:
                target_cls = labels[:, 0]

                # 转换目标框格式
                target_boxes = xywh2xyxy(labels[:, 2:6])
                target_boxes[:, 0] *= img_size[0]
                target_boxes[:, 2] *= img_size[0]
                target_boxes[:, 1] *= img_size[1]
                target_boxes[:, 3] *= img_size[1]

                detected = []
                for *pred_bbox, conf, cls_conf in detections:
                    pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                    # 计算与目标框的IOU
                    iou = bbox_iou(pred_bbox, target_boxes, x1y1x2y2=True)[0]
                    best_i = np.argmax(iou)
                    if iou[best_i] > iou_thres and best_i not in detected:
                        correct.append(1)
                        detected.append(best_i)
                    else:
                        correct.append(0)

            # 计算每个类别的AP
            AP, AP_class, R, P = ap_per_class(
                tp=correct,
                conf=detections[:, 4],
                pred_cls=np.zeros_like(detections[:, 5]),
                target_cls=target_cls
            )

            # 累积AP
            AP_accum_count += np.bincount(AP_class, minlength=nC)
            AP_accum += np.bincount(AP_class, minlength=nC, weights=AP)

            mAPs.append(AP.mean())
            mR.append(R.mean())
            mP.append(P.mean())

            # 计算所有图像的平均值
            mean_mAP = np.sum(mAPs) / (AP_accum_count + 1E-16)
            mean_R = np.sum(mR) / (AP_accum_count + 1E-16)
            mean_P = np.sum(mP) / (AP_accum_count + 1E-16)

        if batch_i % print_interval == 0:
            print(('%11s%11s' + '%11.3g' * 4 + 's') %
                  (seen, dataloader.dataset.nF, mean_P, mean_R, mean_mAP, time.time() - t))

    print('%11s' * 5 % ('Image', 'Total', 'P', 'R', 'mAP'))
    print('AP: %-.4f\n\n' % (AP_accum[0] / (AP_accum_count[0] + 1E-16)))

    return mean_mAP, mean_R, mean_P


def test_emb(
        cfg,
        data_cfg,
        weights,
        batch_size=16,
        iou_thres=0.5,
        conf_thres=0.3,
        nms_thres=0.45,
        print_interval=40,
):
    # Configure run
    f = open(data_cfg)
    data_cfg_dict = json.load(f)
    f.close()
    test_paths = data_cfg_dict['test_emb']
    dataset_root = data_cfg_dict['root']

    # 解析YOLOX配置
    cfg_dict = parse_yolox_cfg(cfg)
    img_size = [cfg_dict['width'], cfg_dict['height']]

    # 初始化YOLOX模型，开启embedding测试模式
    model = YOLOX(cfg_dict, test_emb=True)

    # Load weights
    if weights.endswith('.pt'):
        checkpoint = torch.load(weights, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

    model = torch.nn.DataParallel(model)
    model.cuda().eval()

    # Get dataloader
    transforms = T.Compose([T.ToTensor()])
    dataset = JointDataset(dataset_root, test_paths, img_size, augment=False, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=8,
        drop_last=False,
        collate_fn=collate_fn
    )

    embedding, id_labels = [], []
    print('Extracting pedestrian features...')
    for batch_i, (imgs, targets, paths, shapes, targets_len) in enumerate(dataloader):
        t = time.time()
        output = model(imgs.cuda(), targets.cuda(), targets_len.cuda())

        # 处理YOLOX的embedding输出
        if isinstance(output, dict):
            output = output['embeddings']
        output = output.squeeze()

        for out in output:
            feat, label = out[:-1], out[-1].long()
            if label != -1:
                embedding.append(feat)
                id_labels.append(label)

        if batch_i % print_interval == 0:
            print('Extracting {}/{}, # of instances {}, time {:.2f} sec.'.format(
                batch_i, len(dataloader), len(id_labels), time.time() - t))

    print('Computing pairwise similarity...')
    if len(embedding) < 1:
        return None

    embedding = torch.stack(embedding, dim=0).cuda()
    id_labels = torch.LongTensor(id_labels)
    n = len(id_labels)
    assert len(embedding) == n

    embedding = F.normalize(embedding, dim=1)
    pdist = torch.mm(embedding, embedding.t()).cpu().numpy()
    gt = id_labels.expand(n, n).eq(id_labels.expand(n, n).t()).numpy()

    up_triangle = np.where(np.triu(pdist) - np.eye(n) * pdist != 0)
    pdist = pdist[up_triangle]
    gt = gt[up_triangle]

    far_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    far, tar, threshold = metrics.roc_curve(gt, pdist)
    interp = interpolate.interp1d(far, tar)
    tar_at_far = [interp(x) for x in far_levels]
    for f, fa in enumerate(far_levels):
        print('TPR@FAR={:.7f}: {:.4f}'.format(fa, tar_at_far[f]))
    return tar_at_far


def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    """
    YOLOX后处理函数
    """
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.size(0):
            continue

        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]

        if not detections.size(0):
            continue

        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],
            detections[:, 4] * detections[:, 5],
            detections[:, 6],
            nms_thre,
        )
        detections = detections[nms_out_index]
        output[i] = detections

    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='test.py')
    parser.add_argument('--batch-size', type=int, default=40, help='size of each image batch')
    parser.add_argument('--cfg', type=str, default='cfg/yolox.cfg', help='cfg file path')
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='data config')
    parser.add_argument('--weights', type=str, default='weights/latest.pt', help='path to weights file')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='iou threshold required to qualify as detected')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--nms-thres', type=float, default=0.5, help='iou threshold for non-maximum suppression')
    parser.add_argument('--print-interval', type=int, default=10, help='print interval')
    parser.add_argument('--test-emb', action='store_true', help='test embedding')
    opt = parser.parse_args()
    print(opt, end='\n\n')

    with torch.no_grad():
        if opt.test_emb:
            res = test_emb(
                opt.cfg,
                opt.data_cfg,
                opt.weights,
                opt.batch_size,
                opt.iou_thres,
                opt.conf_thres,
                opt.nms_thres,
                opt.print_interval,
            )
        else:
            mAP = test(
                opt.cfg,
                opt.data_cfg,
                opt.weights,
                opt.batch_size,
                opt.iou_thres,
                opt.conf_thres,
                opt.nms_thres,
                opt.print_interval,
            )