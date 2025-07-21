import torch
import torch.distributed as dist


class ThreshController:
    def __init__(self, nclass, momentum, thresh_init=0.85):

        self.thresh_global = torch.tensor(thresh_init).cuda()
        self.momentum = momentum
        self.nclass = nclass
     #   self.gpu_num = dist.get_world_size()

    def new_global_mask_pooling(self, pred, ignore_mask=None):
        return_dict = {}
        n, c, h, w = pred.shape

        pred = pred
        if ignore_mask is not None:
            ignore_mask = ignore_mask
        mask_pred = torch.argmax(pred, dim=1)
        pred_softmax = pred.softmax(dim=1)
        pred_conf = pred_softmax.max(dim=1)[0]
        unique_cls = torch.unique(mask_pred)
        cls_num = len(unique_cls)
        new_global = 0.0

        for cls in unique_cls:
            cls_map = (mask_pred == cls)

            cls_map = cls_map.unsqueeze(1)



            if ignore_mask is not None:
                cls_map *= (ignore_mask != 255)
            if cls_map.sum() == 0:
                cls_num -= 1
                continue
            cls_map = cls_map.squeeze(1)
            pred_conf_cls_all = pred_conf[cls_map]
            cls_max_conf = pred_conf_cls_all.max()
            new_global += cls_max_conf
        if cls_num > 0:
            return_dict['new_global'] = new_global / cls_num
        else:
            return_dict['new_global'] = None

        return return_dict

    def thresh_update(self, pred, ignore_mask=None, update_g=False):
        thresh = self.new_global_mask_pooling(pred, ignore_mask)

        if isinstance(update_g, torch.Tensor):
            update_g = update_g.any().item()
        if update_g and thresh['new_global'] is not None:
            self.thresh_global = self.momentum * self.thresh_global + (1 - self.momentum) * thresh['new_global']

    def get_thresh_global(self):
        return self.thresh_global
