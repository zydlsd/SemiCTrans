#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
import argparse
from numpy import *
from tqdm import tqdm
import torch.optim as optim
import torch.backends.cudnn as cudnn

from utils.train_utils.trainer import *
from utils.train_utils.losses import contrast_loss

from utils.net_utils.unet import UNet
from utils.net_utils.vision_transformer import SwinUnet
from utils.net_utils.configs.config import get_config
from utils.net_utils.class_contra_net import ProjectHead


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *


def train(args, trainloader, model, loss, optimizers, epoch):
    criterion_dice = loss[0]
    criterion_ce = loss[1]
    model1 = model[0]
    model2 = model[1]
    model3 = model[2]
    optimizer1 = optimizers[0]
    optimizer2 = optimizers[1]
    train_epoch_loss = 0.0
    for _, sampled_batch in enumerate(trainloader):
        volume_batch, label_batch = sampled_batch['preprocess_brain'], sampled_batch['label']
        volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

        outputs1 = model1(volume_batch)
        outputs2 = model2(volume_batch)
        lb_outputs1 = outputs1[:args.labeled_bs]
        lb_outputs2 = outputs2[:args.labeled_bs]
        unlb_outputs1 = outputs1[args.labeled_bs:]
        unlb_outputs2 = outputs2[args.labeled_bs:]

        # L_seg
        loss_sup1 = 0.5 * (criterion_ce(lb_outputs1, label_batch[:args.labeled_bs].long())
                           + criterion_dice(lb_outputs1, label_batch[:args.labeled_bs], softmax=True))
        loss_sup2 = 0.5 * (criterion_ce(lb_outputs2, label_batch[:args.labeled_bs].long())
                           + criterion_dice(lb_outputs2, label_batch[:args.labeled_bs], softmax=True))
        L_seg = loss_sup1 + loss_sup2

        # L_cons
        L_cons, pseudo_label = DUGM(data=volume_batch[args.labeled_bs:],
                                    outputs=[unlb_outputs1, unlb_outputs2],
                                    model=[model1, model2], args=args, T=4, now_epoch=epoch + 1)

        # L_cont
        project_head1 = model3(outputs1)
        project_head2 = model3(outputs2)
        gt = label_batch[:args.labeled_bs]
        L_cont = contrast_loss(args, project_head1, project_head2, gt, pseudo_label)

        # TOTAL LOSS
        Loss = L_seg + 0.6 * L_cons + 0.4 * L_cont

        optimizer1.zero_grad()
        optimizer2.zero_grad()
        Loss.backward()
        optimizer1.step()
        optimizer2.step()

        args.iter_num += 1
        train_epoch_loss += Loss.item()

        # change lr
        if epoch % args.change_lr == 0:
            lr_ = args.base_lr1 * 0.1 ** (epoch // args.change_lr)
            for param_group in optimizer1.param_groups:
                param_group['lr'] = lr_

        if epoch % args.change_lr == 0:
            lr_ = args.base_lr2 * 0.1 ** (epoch // args.change_lr)
            for param_group in optimizer2.param_groups:
                param_group['lr'] = lr_

        # save train
        if Loss.item() < args.best_train_loss:
            args.best_train_loss = Loss.item()
            torch.save({"state_dict": model1.state_dict(),
                        "optimizer": optimizer1.state_dict()},
                       os.path.join(args.save_path, 'best_train_u.pth'))
            torch.save({"state_dict": model2.state_dict(),
                        "optimizer": optimizer2.state_dict()},
                       os.path.join(args.save_path, 'best_train_s.pth'))

        if args.iter_num % 50 == 0:
            torch.save({"state_dict": model1.state_dict(),
                        "optimizer": optimizer1.state_dict()},
                       os.path.join(args.save_path, 'iter_{}_train_u.pth'.format(args.iter_num)))
            torch.save({"state_dict": model2.state_dict(),
                        "optimizer": optimizer2.state_dict()},
                       os.path.join(args.save_path, 'iter_{}_train_s.pth'.format(args.iter_num)))

            print('[epoch|iter]:[%d|%d] [lr_1:%.6f lr_2:%.6f] Loss:%.4f L_seg:%.4f L_cons:%.4f L_cont:%.4f best_train_loss:%.4f' %
                  (epoch + 1, args.iter_num,
                   optimizer1.param_groups[0]['lr'],
                   optimizer2.param_groups[0]['lr'],
                   Loss.item(), L_seg.item(), L_cons.item(), L_cont.item(),
                   args.best_train_loss))

    train_epoch_loss /= len(trainloader)
    return train_epoch_loss


def run_main(args):
    model1 = UNet(in_chns=args.net_in_chns, class_num=args.num_classes)
    assert args.crop_size[0] % 32 == 0
    model2 = SwinUnet(config, img_size=args.crop_size[0], num_classes=args.num_classes).cuda()
    model3 = ProjectHead(class_num=args.num_classes)

    model1.cuda()
    model2.cuda()
    model3.cuda()

    trainloader = data_load(args)

    model1.train()
    model2.train()

    optimizer1 = optim.SGD(model1.parameters(), lr=args.base_lr1, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(model2.parameters(), lr=args.base_lr2, momentum=0.9, weight_decay=0.0001)
    optimizers = [optimizer1, optimizer2]

    criterion_dice, criterion_ce = train_sup_loss(args)
    loss = [criterion_dice, criterion_ce]

    model = [model1, model2, model3]

    args.max_iterations = args.max_epochs * len(trainloader)

    for epoch in tqdm(range(args.max_epochs), ncols=70):
        train(args, trainloader, model, loss, optimizers, epoch)
        if args.iter_num >= args.max_iterations:
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--net_in_chns', type=int, default=12)
    parser.add_argument('--num_classes', type=int, default=4, help='output channel of network')
    parser.add_argument('--crop_size', type=list, default=(224, 224), help='patch_size')
    parser.add_argument('--calculated_weight_train_data', type=bool, default=True)

    parser.add_argument('--train_data_num', type=int, default=300)
    parser.add_argument('--base_lr1', type=float, default=0.01, help='unet')
    parser.add_argument('--base_lr2', type=float, default=0.01, help='swinunet')
    parser.add_argument('--change_lr', type=int, default=150)
    parser.add_argument('--max_epochs', type=int, default=400)

    parser.add_argument('--consistency', type=float, default=0.1)
    parser.add_argument('--consistency_rampup', type=int, default=300)
    parser.add_argument('--contrast_temperature', type=float, default=1)

    parser.add_argument('--batch_size', type=int, default=24)
    parser.add_argument('--labeled_bs', type=int, default=12)

    parser.add_argument('--data_path', type=str, default='../SemiCTrans/src/data/Brats2020Train', help='user-defined')

    parser.add_argument('--save_path', type=str, default='../SemiCTrans/train_out', help='user-defined')
    parser.add_argument('--label_rate', type=float, default=0.2, help='0.01, 0.1, 0.2,..,1')
    parser.add_argument('--gpu_name', type=str, default='1', help='user-defined')

    # SwinUnet
    parser.add_argument('--cfg', type=str, default="./utils/net_utils/configs/swin_tiny_patch4_window7_224_lite.yaml", help='user-defined')
    parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache_mode', type=str, default='part', choices=['no', 'full', 'part'])
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true', help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'], help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    args = parser.parse_args()
    config = get_config(args)

    # set gpu
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_name

    # maintain reproducibility
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    print('torch._version', torch.__version__)

    # set savefile
    now = time.localtime()
    now_format = time.strftime("%Y-%m-%d %H:%M:%S", now)
    date_now, time_now = now_format.split(' ')
    date_now = date_now.replace("-", "")
    time_now = time_now.replace(":", "")
    args.data_label = math.ceil(args.label_rate * args.train_data_num)
    args.exp_name = f"{date_now}_{time_now}_label{args.data_label}"
    args.save_path = os.path.join(args.save_path, args.exp_name)
    os.makedirs(args.save_path, exist_ok=True)

    # define global variable parameter initialization
    args.iter_num = 0
    args.best_train_loss = 1e5

    args.loss_weight = None
    if args.calculated_weight_train_data:
        args.loss_weight = calculated_weight_train_data(args)
    args.dice_weight = args.ce_weight = args.contrast_weight = args.loss_weight

    run_main(args)
