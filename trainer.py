from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable
import torch.nn.functional as F

from meters import AverageMeter


def iou(heat_map, label, average=True):
    # output and label are both of dimension [n,c,h,w]
    onehot_label = onehot(label, 2)
    _, pred = torch.max(heat_map, dim=1, keepdim=True)
    onehot_pred = onehot(pred, 2)
    # inters = (pred, onehot_label).type(torch.cuda.FloatTensor if is_cuda else torch.FloatTensor)
    n, c, h, w = onehot_label.size()
    onehot_pred = onehot_pred.view(n, c, h * w).byte()
    onehot_label = onehot_label.view(n, c, h * w).byte()
    inters = (onehot_pred & onehot_label).float().sum(dim=2)
    unions = (onehot_pred | onehot_label).float().sum(dim=2)
    # unions = (pred & 1).sum(dim=2).float() + (onehot_label & 1).sum(dim=2).float() - inters
    if average:
        iou_ = (inters / unions).mean()
    else:
        iou_ = (inters / unions).mean(dim=1)
    return iou_


def summary_output_lbl(output, label, writer, step, interval=5):
    if (step + 1) % interval == 0:
        n = output.size(0)
        onehot_label = onehot(label, 2)[0]

        _, pred = torch.max(output, dim=1, keepdim=True)
        onehot_pred = onehot(pred, 2)[0]

        a = torch.unsqueeze(onehot_pred, dim=1)
        b = torch.unsqueeze(onehot_label, dim=1)

        writer.add_image('pred', a, step)
        writer.add_image('lbl', b, step)


def onehot(x, l):
    n, c, h, w = x.size()
    assert c == 1, 'segmentation is not single channel'
    res = torch.zeros(n, l, h, w)
    if x.is_cuda:
        res = res.cuda()
    # label = label.view(32, 1, -1).cpu()
    # onehot_label = onehot_label.view(32, 2, -1).cpu()

    res = res.scatter_(dim=1, index=x, value=1.)
    return res


# class BaseTrainer(object):
#     def __init__(self, sn, vn, criterion_sn, criterion_vn):
#         super(BaseTrainer, self).__init__()
#         self.sn = sn
#         self.vn = vn
#         self.criterion_sn = criterion_sn
#         self.criterion_vn = criterion_vn
#
#     def train(self, epoch, data_loader, optimizer, writer=None, print_freq=1):
#         raise NotImplementedError
#
#     # self.model.train()
#     #
#     # batch_time = AverageMeter()
#     # data_time = AverageMeter()
#     # losses = AverageMeter()
#     # ious = AverageMeter()
#     #
#     # end = time.time()
#     # for i, inputs in enumerate(data_loader):
#     #     data_time.update(time.time() - end)
#     #
#     #     img, lbl = self._parse_data(inputs)
#     #     loss, iou_, outputs = self._forward(img, lbl)
#     #
#     #     losses.update(loss.data[0], lbl.size(0))
#     #     ious.update(iou_, lbl.size(0))
#     #
#     #     # bp % gd
#     #     optimizer.zero_grad()
#     #     loss.backward()
#     #     optimizer.step()
#     #
#     #     batch_time.update(time.time() - end)
#     #     end = time.time()
#     #
#     #     if (i + 1) % print_freq == 0:
#     #         print('Epoch: [{}][{}/{}]\t'
#     #               'Time {:.3f} ({:.3f})\t'
#     #               'Data {:.3f} ({:.3f})\t'
#     #               'Loss {:.3f} ({:.3f})\t'
#     #               'Prec {:.2%} ({:.2%})\t'
#     #               .format(epoch, i + 1, len(data_loader),
#     #                       batch_time.val, batch_time.avg,
#     #                       data_time.val, data_time.avg,
#     #                       losses.val, losses.avg,
#     #                       ious.val, ious.avg))
#
#     def _parse_data(self, inputs):
#         raise NotImplementedError
#
#     def _forward(self, inputs, targets):
#         raise NotImplementedError


class Trainer:
    def __init__(self, sn, vn, criterion_sn, criterion_vn):
        # super(BaseTrainer, self).__init__()
        self.sn = sn
        self.vn = vn
        self.criterion_sn = criterion_sn
        self.criterion_vn = criterion_vn

    def _parse_data(self, inputs):
        _, _, img, lbl = inputs
        img = Variable(img.cuda())
        lbl = Variable(lbl.cuda())
        return img, lbl

    def _forward_sn(self, inputs, targets):
        if not isinstance(inputs, list):
            inputs = [inputs]
        outputs = F.softmax(self.sn(*inputs), dim=1)

        if isinstance(self.criterion_sn, torch.nn.NLLLoss2d):
            loss = self.criterion_sn(outputs, targets.squeeze())
            iou_ = iou(outputs.data, targets.data)
        else:
            raise ValueError("Unsupported loss:", self.criterion_sn)
        return loss, iou_, outputs

    def _forward_vn(self, img, lbl_fake, target_iou):
        iou_pred = self.vn(img, lbl_fake)  # iou_pred

        if isinstance(self.criterion_vn, torch.nn.MSELoss):
            loss = self.criterion_vn(iou_pred, Variable(target_iou))
            # loss = torch.pow(iou_pred - Variable(target_iou), 2.).mean()
        else:
            raise ValueError("Unsupported loss:", self.criterion_vn)
        return loss, iou_pred

    def step(self, opt, loss):
        opt.zero_grad()
        loss.backward()
        opt.step()

    def train(self, epoch, data_loader, opt_sn, opt_vn, mode, writer=None, print_freq=1):
        self.sn.train()
        self.vn.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses_sn = AverageMeter()
        losses_vn = AverageMeter()
        ious = AverageMeter()

        end = time.time()

        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            img, lbl = self._parse_data(inputs)

            # train sn
            loss_sn, iou_, heat_map = self._forward_sn(img, lbl)
            losses_sn.update(loss_sn.data[0], lbl.size(0))
            ious.update(iou_, lbl.size(0))

            if mode == 'sn':
                # if opt_sn is None:
                #     img.volatile = True
                #     lbl.volatile = True
                # else:
                #     img.volatile = False
                #     lbl.volatile = False

                self.step(opt_sn, loss_sn)
            # train vn
            elif mode == 'vn':
                # heat_map = heat_map.detach()
                _, seg_pred = torch.max(heat_map, dim=1, keepdim=True)
                # seg_pred = onehot(seg_pred, 2)
                # heat_map = heat_map
                target_iou = iou(heat_map.data, lbl.data, average=False)

                loss_vn, iou_pred = self._forward_vn(img, heat_map, target_iou)
                losses_vn.update(loss_vn.data[0], lbl.size(0))
                self.step(opt_vn, loss_vn)

            # bp % gd
            # if opt_sn is not None:
            #     self.step(opt_sn, loss_sn)
            # if opt_vn is not None:
            #     self.step(opt_vn, loss_vn)

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss_sn {:.3f} ({:.3f})\t'
                      'Loss_vn {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses_sn.val, losses_sn.avg,
                              losses_vn.val, losses_vn.avg,
                              ious.val, ious.avg))

        if writer is not None:
            summary_output_lbl(seg_pred.data, lbl.data, writer, epoch)
