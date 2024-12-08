import argparse
import json
import time
from collections import defaultdict
from time import gmtime, strftime

from torch import nn

import test
from Yolox import YOLOX, CSPDarknet, YOLOXHead  # 替换为YOLOX相关导入
from shutil import copyfile
from utils.datasets import JointDataset, collate_fn
from utils.utils import *
from utils.log import logger
from torchvision.transforms import transforms as T
from torch.cuda import amp  # 添加混合精度训练支持


def train(
        cfg,
        data_cfg,
        weights_from="",
        weights_to="",
        save_every=10,
        img_size=(640, 640),  # YOLOX默认使用640x640
        resume=False,
        epochs=100,
        batch_size=16,
        accumulated_batches=1,
        freeze_backbone=False,
        opt=None,
):
    # 配置训练时间戳
    timme = strftime("%Y-%d-%m %H:%M:%S", gmtime())
    timme = timme[5:-3].replace('-', '_')
    timme = timme.replace(' ', '_')
    timme = timme.replace(':', '_')
    weights_to = osp.join(weights_to, 'run' + timme)
    mkdir_if_missing(weights_to)

    if resume:
        latest_resume = osp.join(weights_from, 'latest.pt')

    torch.backends.cudnn.benchmark = True

    # 配置数据
    f = open(data_cfg)
    data_config = json.load(f)
    trainset_paths = data_config['train']
    dataset_root = data_config['root']
    f.close()

    # 数据增强转换
    transforms = T.Compose([
        T.ToTensor(),
        # 可以添加YOLOX特有的数据增强
    ])

    # 获取数据加载器
    dataset = JointDataset(dataset_root, trainset_paths, img_size, augment=True, transforms=transforms)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn
    )

    # 初始化YOLOX模型
    backbone = CSPDarknet(
        dep_mul=1.0,
        wid_mul=1.0,
        nID=dataset.nID  # 保持ReID支持
    )
    head = YOLOXHead(num_classes=dataset.nC)  # 使用数据集的类别数
    model = YOLOX(backbone=backbone, head=head, nID=dataset.nID)

    # 读取预训练权重或恢复训练
    if resume:
        checkpoint = torch.load(latest_resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        start_epoch = checkpoint['epoch'] + 1
    else:
        start_epoch = 0

    model.cuda().train()

    # 配置优化器
    pg0, pg1, pg2 = [], [], []  # 参数分组优化
    for k, v in model.named_modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
            pg2.append(v.bias)
        if isinstance(v, nn.BatchNorm2d) or "bn" in k:
            pg0.append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
            pg1.append(v.weight)

    optimizer = torch.optim.SGD(pg0, lr=opt.lr, momentum=0.9, nesterov=True)
    optimizer.add_param_group({"params": pg1, "weight_decay": 5e-4})
    optimizer.add_param_group({"params": pg2})

    if resume and checkpoint['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])

    # 使用YOLOX的学习率策略
    max_epoch = epochs
    min_lr_ratio = 0.05
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, epochs, eta_min=opt.lr * min_lr_ratio
    )

    model = torch.nn.DataParallel(model)
    scaler = amp.GradScaler(enabled=True)

    # 训练循环
    t0 = time.time()
    for epoch in range(start_epoch, epochs):
        model.train()
        logger.info(('%8s%12s' + '%10s' * 6) % (
            'Epoch', 'Batch', 'box', 'conf', 'cls', 'id', 'total', 'time'))

        if freeze_backbone and epoch < 2:
            for name, p in model.named_parameters():
                if "backbone" in name:
                    p.requires_grad = False if (epoch == 0) else True

        ui = -1
        rloss = defaultdict(float)
        optimizer.zero_grad()

        for i, (imgs, targets, _, _, targets_len) in enumerate(dataloader):
            ni = i + len(dataloader) * epoch
            imgs = imgs.cuda()
            targets = targets.cuda()
            targets_len = targets_len.cuda()

            # 混合精度训练
            with amp.autocast():
                outputs = model(imgs, targets, targets_len)
                loss = outputs["total_loss"]
                loss_box = outputs["iou_loss"]
                loss_conf = outputs["conf_loss"]
                loss_cls = outputs["cls_loss"]
                loss_id = outputs.get("id_loss", torch.zeros(1).cuda())

            # 优化器步骤
            scaler.scale(loss).backward()
            if ((i + 1) % accumulated_batches == 0) or (i == len(dataloader) - 1):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # 更新运行损失
            ui += 1
            rloss['box'] = (rloss['box'] * ui + float(loss_box)) / (ui + 1)
            rloss['conf'] = (rloss['conf'] * ui + float(loss_conf)) / (ui + 1)
            rloss['cls'] = (rloss['cls'] * ui + float(loss_cls)) / (ui + 1)
            rloss['id'] = (rloss['id'] * ui + float(loss_id)) / (ui + 1)
            rloss['loss'] = (rloss['loss'] * ui + float(loss)) / (ui + 1)

            # 打印训练信息
            if i % opt.print_interval == 0:
                logger.info(('%8s%12s' + '%10.3g' * 6) % (
                    f'{epoch}/{epochs - 1}',
                    f'{i}/{len(dataloader) - 1}',
                    rloss['box'],
                    rloss['conf'],
                    rloss['cls'],
                    rloss['id'],
                    rloss['loss'],
                    time.time() - t0
                ))
                t0 = time.time()

        # 保存检查点
        checkpoint = {
            'epoch': epoch,
            'model': model.module.state_dict(),
            'optimizer': optimizer.state_dict()
        }

        copyfile(cfg, weights_to + '/cfg/yolox.cfg')
        copyfile(data_cfg, weights_to + '/cfg/ccmcpe.json')

        latest = osp.join(weights_to, 'latest.pt')
        torch.save(checkpoint, latest)

        if epoch % save_every == 0 and epoch != 0:
            checkpoint["optimizer"] = []
            torch.save(checkpoint, osp.join(weights_to, f"weights_epoch_{epoch}.pt"))

        # 验证
        if epoch % opt.test_interval == 0:
            with torch.no_grad():
                mAP, R, P = test.test(cfg, data_cfg, weights=latest, batch_size=batch_size,
                                      img_size=img_size, print_interval=40, nID=dataset.nID)
                test.test_emb(cfg, data_cfg, weights=latest, batch_size=batch_size,
                              img_size=img_size, print_interval=40, nID=dataset.nID)

        scheduler.step()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=300, help='number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
    parser.add_argument('--accumulated-batches', type=int, default=1, help='number of batches before optimizer step')
    parser.add_argument('--cfg', type=str, default='cfg/yolox.cfg', help='cfg file path')
    parser.add_argument('--weights-from', type=str, default='weights/')
    parser.add_argument('--weights-to', type=str, default='weights/')
    parser.add_argument('--save-model-after', type=int, default=10)
    parser.add_argument('--data-cfg', type=str, default='cfg/ccmcpe.json', help='data config file path')
    parser.add_argument('--img-size', type=int, default=[640, 640], nargs='+', help='pixels')
    parser.add_argument('--resume', action='store_true', help='resume training flag')
    parser.add_argument('--print-interval', type=int, default=40, help='print interval')
    parser.add_argument('--test-interval', type=int, default=10, help='test interval')
    parser.add_argument('--lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument('--unfreeze-bn', action='store_true', help='unfreeze bn')
    opt = parser.parse_args()

    init_seeds()

    train(
        opt.cfg,
        opt.data_cfg,
        weights_from=opt.weights_from,
        weights_to=opt.weights_to,
        save_every=opt.save_model_after,
        img_size=opt.img_size,
        resume=opt.resume,
        epochs=opt.epochs,
        batch_size=opt.batch_size,
        accumulated_batches=opt.accumulated_batches,
        opt=opt,
    )