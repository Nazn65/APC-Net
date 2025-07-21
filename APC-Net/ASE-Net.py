import argparse
import logging
import os
import pprint
from tqdm import tqdm
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml
from einops import rearrange

from dataset.busi import BUSIDataset
from dataset.semi import SemiDataset
from model.unet import UNet
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.thresh_helper import ThreshController
from evaluate import evaluate
import numpy as np
import cv2
import matplotlib.pyplot as plt
parser = argparse.ArgumentParser(
    description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, default=r'./configs/busi.yaml',
                    help='path to config file')
parser.add_argument('--labeled-id-path', type=str,
                    default=r'./splits/BUSI/1/label.txt',
                    help='Path to labeled')
parser.add_argument('--unlabeled-id-path', type=str,
                    default=r'./splits/BSUI/1/unlabel.txt',
                    help='Path to unlabeled')
parser.add_argument('--save-path', type=str, default=r'./save',
                    help='Path to save')
parser.add_argument('--local_rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
args = parser.parse_args()



import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r", encoding='utf-8'), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank = 0
    world_size = 1


    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))

        writer = SummaryWriter(args.save_path)

        os.makedirs(args.save_path, exist_ok=True)

    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet(in_chns=1, class_num=cfg['nclass'],cfg={**cfg,'pretrained':False})
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))

    optimizer = SGD(model.parameters(), lr=cfg['lr'], momentum=0.9, weight_decay=0.0001)
    model.cuda()


    criterion_ce = nn.CrossEntropyLoss(ignore_index=255)

    criterion_dice = DiceLoss(n_classes=cfg['nclass'])

    trainset_u = BUSIDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = BUSIDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = BUSIDataset(cfg['dataset'], cfg['data_root'], 'val')

    # trainset_u = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_u',
    #                          cfg['crop_size'], args.unlabeled_id_path)
    #
    # trainset_l = SemiDataset(cfg['dataset'], cfg['data_root'], 'train_l',
    #                          cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    # valset = SemiDataset(cfg['dataset'], cfg['data_root'], 'val')

    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)

    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=1, drop_last=True)

    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=1, drop_last=True)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False)

    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1

    thresh_controller = ThreshController(nclass=4, momentum=0.999, thresh_init=cfg['thresh_init'])
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'),weights_only=True)
        model.load_state_dict(checkpoint['model'], strict=False)

        previous_best = checkpoint['previous_best']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)

    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_mask_ratio = AverageMeter()
        total_loss_corr_ce = AverageMeter()
        total_loss_corr_u = AverageMeter()
        total_loss_dice=AverageMeter()



        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        if rank == 0:
            tbar = tqdm(total=len(trainloader_l))

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2,  ignore_mask, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, ignore_mask_mix, _,  _)) in enumerate(loader):

            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2, ignore_mask = img_u_s1.cuda(), img_u_s2.cuda(), ignore_mask.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix, ignore_mask_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda(), ignore_mask_mix.cuda()
            b, c, h, w = img_x.shape

            with torch.no_grad():

                model.eval()
                res_u_w_mix = model(img_u_w_mix, need_fp=False, use_corr=False)
                pred_u_w_mix = res_u_w_mix['out'].detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]
            res_w = model(torch.cat((img_x, img_u_w)), need_fp=True, use_corr=True)
            preds = res_w['out']
            preds_fp = res_w['out_fp']
            preds_corr = res_w['corr_out']
            preds_corr_map = res_w['corr_map']

            pred_x_corr, pred_u_w_corr = preds_corr.split([num_lb, num_ulb])
            pred_u_w_corr_map = preds_corr_map[num_lb:]
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]
            res_s1 = model(img_u_s1, need_fp=False, use_corr=True)
            pred_u_s1 = res_s1['out']
            pred_u_s1_corr = res_s1['corr_out']
            res_s2 = model(img_u_s2, need_fp=False, use_corr=True)
            pred_u_s2 = res_s2['out']
            pred_u_s2_corr = res_s2['corr_out']
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1  = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed1 = rearrange(pred_u_w_corr_map.clone(),'n h w -> n 1 h w')
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed1.shape

            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2  = \
                mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            corr_map_u_w_cutmixed2 = rearrange(pred_u_w_corr_map.clone(), 'n h w -> n 1 h w')
            b_sample, c_sample, _, _ = corr_map_u_w_cutmixed2.shape

            cutmix_box1_map = (cutmix_box1 == 1)
            cutmix_box2_map = (cutmix_box2 == 1)
            mask_u_w_cutmixed1[cutmix_box1_map] = mask_u_w_mix[cutmix_box1_map]
            mask_u_w_cutmixed1_copy = mask_u_w_cutmixed1.clone()
            conf_u_w_cutmixed1[cutmix_box1_map] = conf_u_w_mix[cutmix_box1_map]

            mask_u_w_cutmixed2[cutmix_box2_map] = mask_u_w_mix[cutmix_box2_map]
            mask_u_w_cutmixed2_copy = mask_u_w_cutmixed2.clone()
            conf_u_w_cutmixed2[cutmix_box2_map] = conf_u_w_mix[cutmix_box2_map]

            ignore_mask_mix = ignore_mask_mix
            cutmix_box1_sample = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            cutmix_box2_sample = rearrange(cutmix_box2_map, 'n h w -> n 1 h w')
            cutmix_box1_map = rearrange(cutmix_box1_map, 'n h w -> n 1 h w')
            cutmix_box2_map = rearrange(cutmix_box2_map, 'n h w -> n 1 h w')

            ignore_mask_cutmixed1[cutmix_box1_map] = ignore_mask_mix[cutmix_box1_map]
            ignore_mask_cutmixed1_sample = ignore_mask_cutmixed1 != 255

            ignore_mask_cutmixed2[cutmix_box2_map] = ignore_mask_mix[cutmix_box2_map]
            ignore_mask_cutmixed2_sample = ignore_mask_cutmixed2 != 255

            corr_map_u_w_cutmixed1 = (
                        corr_map_u_w_cutmixed1 * ~cutmix_box1_sample * ignore_mask_cutmixed1_sample).bool()

            corr_map_u_w_cutmixed2 = (
                    corr_map_u_w_cutmixed2 * ~cutmix_box2_sample * ignore_mask_cutmixed2_sample).bool()

            thresh_controller.thresh_update(pred_u_w.detach(), ignore_mask_cutmixed1, ignore_mask_cutmixed2)
            thresh_global = thresh_controller.get_thresh_global()


            conf_fliter_u_w = ((conf_u_w_cutmixed1 >= thresh_global) & (conf_u_w_cutmixed2 >= thresh_global) &(ignore_mask_cutmixed1 != 255)&(ignore_mask_cutmixed2 != 255))
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w.clone()
            conf_fliter_u_w_sample = conf_fliter_u_w_without_cutmix

            segments = (corr_map_u_w_cutmixed2 * conf_fliter_u_w_sample * corr_map_u_w_cutmixed1 * conf_fliter_u_w_sample).bool()


            for img_idx in range(b_sample):
                for segment_idx in range(c_sample):
                    segment = segments[img_idx, segment_idx]
                    segment_ori_1 = corr_map_u_w_cutmixed1[img_idx, segment_idx]
                    segment_ori_2 = corr_map_u_w_cutmixed2[img_idx, segment_idx]
                    high_conf_ratio_1 = torch.sum(segment) / torch.sum(segment_ori_1)
                    high_conf_ratio_2 = torch.sum(segment) / torch.sum(segment_ori_2)

                    if torch.sum(segment) == 0 or high_conf_ratio_1 < thresh_global or high_conf_ratio_2 < thresh_global:
                        continue

                    unique_cls_1, count_1 = torch.unique(mask_u_w_cutmixed1[img_idx][segment == 1], return_counts=True)

                    unique_cls_2, count_2 = torch.unique(mask_u_w_cutmixed2[img_idx][segment == 1], return_counts=True)

                    if torch.max(count_1) / torch.sum(count_1) > thresh_global:
                        top_class_1 = unique_cls_1[torch.argmax(count_1)]
                        mask_u_w_cutmixed1[img_idx][segment_ori_1 == 1] = top_class_1

                    if torch.max(count_2) / torch.sum(count_2) > thresh_global:
                        top_class_2 = unique_cls_2[torch.argmax(count_2)]
                        mask_u_w_cutmixed2[img_idx][segment_ori_2 == 1] = top_class_2

                        conf_fliter_u_w_without_cutmix[img_idx] = (
                                conf_fliter_u_w_without_cutmix[img_idx] | segment_ori_1 | segment_ori_2
                        )
            conf_fliter_u_w_without_cutmix = conf_fliter_u_w_without_cutmix | conf_fliter_u_w

            loss_x = (criterion_ce(pred_x, mask_x)+criterion_dice(pred_x.softmax(dim=1),mask_x.unsqueeze(1).float())) / 2.0
            loss_x_corr = (criterion_ce(pred_x_corr, mask_x)+criterion_dice(pred_x_corr.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1 = (criterion_ce(pred_u_s1, mask_u_w_cutmixed1)+criterion_dice(pred_u_s1.softmax(dim=1),mask_u_w_cutmixed1.unsqueeze(1).float())) / 2.0
            loss_u_s1 = loss_u_s1 * conf_fliter_u_w_without_cutmix


            loss_u_s1 = torch.sum(loss_u_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_s2 =( criterion_ce(pred_u_s2, mask_u_w_cutmixed2)+criterion_dice(pred_u_s2.softmax(dim=1),mask_u_w_cutmixed2.unsqueeze(1).float()))/ 2.0
            loss_u_s2 = loss_u_s2 * conf_fliter_u_w_without_cutmix
            loss_u_s2 = torch.sum(loss_u_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()

            loss_u_corr_s1 = (criterion_ce(pred_u_s1_corr, mask_u_w_cutmixed1)+criterion_dice(pred_u_s1_corr.softmax(dim=1),mask_u_w_cutmixed1.unsqueeze(1).float()))/ 2.0
            loss_u_corr_s1 = loss_u_corr_s1 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s1 = torch.sum(loss_u_corr_s1) / torch.sum(ignore_mask_cutmixed1 != 255).item()

            loss_u_corr_s2 = (criterion_ce(pred_u_s2_corr, mask_u_w_cutmixed2)+criterion_dice(pred_u_s2_corr.softmax(dim=1),mask_u_w_cutmixed2.unsqueeze(1).float()))/ 2.0
            loss_u_corr_s2 = loss_u_corr_s2 * conf_fliter_u_w_without_cutmix
            loss_u_corr_s2 = torch.sum(loss_u_corr_s2) / torch.sum(ignore_mask_cutmixed2 != 255).item()

            loss_u_corr_s = loss_u_corr_s1 + loss_u_corr_s2

            loss_u_corr_w = (criterion_ce(pred_u_w_corr, mask_u_w)+criterion_dice(pred_u_w_corr.softmax(dim=1),mask_u_w.unsqueeze(1).float()))/ 2.0
            loss_u_corr_w = loss_u_corr_w * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_corr_w = torch.sum(loss_u_corr_w) / torch.sum(ignore_mask != 255).item()
            loss_u_corr = 0.5 * (loss_u_corr_s + loss_u_corr_w)

            softmax_pred_u_w = F.softmax(pred_u_w.detach(), dim=1)
            logsoftmax_pred_u_s1 = F.log_softmax(pred_u_s1, dim=1)
            logsoftmax_pred_u_s2 = F.log_softmax(pred_u_s2, dim=1)



            loss_u_w_fp = (criterion_ce(pred_u_w_fp, mask_u_w)+criterion_dice(pred_u_w_fp.softmax(dim=1),mask_u_w.unsqueeze(1).float()))/ 2.0
            loss_u_w_fp = loss_u_w_fp * ((conf_u_w >= thresh_global) & (ignore_mask != 255))
            loss_u_w_fp = torch.sum(loss_u_w_fp) / torch.sum(ignore_mask != 255).item()

            loss = (0.5*loss_x + 0.5 * loss_x_corr + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.25 + 0.25 * loss_u_corr) / 2.0

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            # total_loss_dice += loss_u_dice.item()
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_loss_corr_ce.update(loss_x_corr.item())
            total_loss_corr_u.update(loss_u_corr.item())
            total_mask_ratio.update(((conf_u_w >= thresh_global) & (ignore_mask != 255)).sum().item() / \
                                    (ignore_mask != 255).sum().item())

            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr

            if rank == 0:
                writer.add_scalar('train/total_loss', total_loss.avg, iters)
                writer.add_scalar('train/total_loss_x', total_loss_x.avg, iters)
                writer.add_scalar('train/total_loss_corr_ce',total_loss_corr_ce.avg, iters)
                writer.add_scalar('train/total_loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/total_loss_w_fp', total_loss_w_fp.avg, iters)
                writer.add_scalar('train/total_mask_ratio', total_mask_ratio.avg, iters)
                writer.add_scalar('train/total_loss_corr_u', total_loss_corr_u.avg, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info(
                    'Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f},Loss corr_ce: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Mask ratio: '
                    '{:.3f}, Loss corr_u:{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_corr_ce.avg, total_loss_s.avg,
                                    total_loss_w_fp.avg, total_mask_ratio.avg, total_loss_corr_u.avg))
        model.eval()

        inter = ((pred_x == 0) * (mask_x == 0)).sum().item()
        union = (pred_x == 0).sum().item() + (mask_x == 0).sum().item()
        dice_score = 2* inter / union


        if rank == 0:
            logger.info('***** Evaluation ***** >>>>  Dice: '
                        '{:.2f}'.format(dice_score))

            writer.add_scalar('eval/Dice', dice_score, epoch)
        is_best = dice_score > previous_best
        previous_best = max(dice_score, previous_best)

        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
