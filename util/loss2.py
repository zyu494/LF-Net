# loss function for train the model
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
os.environ['CUDA_LAUNCH_BLOCKING']='1'

def Loss(prob, gt_dict, aux_pred=None, aux_gt=None):
        """
            pred_dict: the dictionary containing model prediction,
                {
                    "target_prob":  the predicted probability of each target candidate,
                    "offset":       the predicted offset of the target position from the gt target candidate,
                    "traj_with_gt": the predicted trajectory with the gt target position as the input,
                    "traj":         the predicted trajectory without the gt target position,
                    "score":        the predicted score for each predicted trajectory,
                }
            gt_dict: the dictionary containing the prediction gt,
                {
                    "target_prob":  the one-hot gt of traget candidate;
                    "offset":       the gt for the offset of the nearest target candidate to the target position;
                    "y":            the gt trajectory of the target agent;
                }
        """
        
        batch_size = prob.size()[0]
        loss = 0.0

        
        cls_loss = F.binary_cross_entropy(
            prob, gt_dict['target_prob'].float(), reduction='none')

        gt_idx = gt_dict['target_prob'].nonzero()
        check_list=[]
        for i in range (batch_size):
             cal=sum(gt_dict['target_prob'][i])
             if cal==torch.tensor(0):
                  check_list.append(i)

        
        cls_loss = cls_loss.sum()
        
      
        loss += cls_loss
        #loss+=cls_loss

        
        #loss_dict = {"tar_cls_loss": cls_loss, "tar_offset_loss": offset_loss}

        return loss#, loss_dict

class TNTLoss(nn.Module):
    """
        The loss function for train TNT, loss = a1 * Targe_pred_loss + a2 * Traj_reg_loss + a3 * Score_loss
    """
    def __init__(self,
                 #lambda1,
                 #m,
                 #k,
                 temper=0.01,
                 aux_loss=False,
                 reduction='sum',
                 device=torch.device("cpu")):
        """
        lambda1, lambda2, lambda3: the loss coefficient;
        temper: the temperature for computing the score gt;
        aux_loss: with the auxiliary loss or not;
        reduction: loss reduction, "sum" or "mean" (batch mean);
        """
        super(TNTLoss, self).__init__()
        #self.lambda1 = lambda1
        # self.lambda2 = lambda2
        # self.lambda3 = lambda3

        # self.m = m
        # self.k = k

        self.aux_loss = aux_loss
        self.reduction = reduction
        self.temper = temper

        self.device = device

    def forward(self, pred_dict, gt_dict, aux_pred=None, aux_gt=None):
        """
            pred_dict: the dictionary containing model prediction,
                {
                    "target_prob":  the predicted probability of each target candidate,
                    "offset":       the predicted offset of the target position from the gt target candidate,
                    "traj_with_gt": the predicted trajectory with the gt target position as the input,
                    "traj":         the predicted trajectory without the gt target position,
                    "score":        the predicted score for each predicted trajectory,
                }
            gt_dict: the dictionary containing the prediction gt,
                {
                    "target_prob":  the one-hot gt of traget candidate;
                    "offset":       the gt for the offset of the nearest target candidate to the target position;
                    "y":            the gt trajectory of the target agent;
                }
        """
        batch_size = pred_dict['target_prob'].size()[0]
        loss = 0.0

        # compute target prediction loss
        # weight = torch.tensor([1.0, 2.0], dtype=torch.float, device=self.device)
        # cls_loss = F.cross_entropy(
        #     pred_dict['target_prob'].transpose(1, 2),
        #     gt_dict['target_prob'].long(),
        #     weight=weight,
        #     reduction='sum')
        # cls_loss = F.binary_cross_entropy_with_logits(
        cls_loss = F.binary_cross_entropy(
            pred_dict['target_prob'], gt_dict['target_prob'].float(), reduction='none')

        gt_idx = gt_dict['target_prob'].nonzero()
        offset = pred_dict['offset'][gt_idx[:, 0], gt_idx[:, 1]]

        # cls_loss, indices = torch.topk(cls_loss, self.m, dim=1)    # largest 50
        cls_loss = cls_loss.sum()
        offset_loss = F.smooth_l1_loss(offset, gt_dict['offset'], reduction='sum')
        # loss += self.lambda1 * (cls_loss + offset_loss) / (1.0 if self.reduction == "sum" else batch_size)
        loss += self.lambda1 * (cls_loss + offset_loss)

        # # compute motion estimation loss
        # reg_loss = F.smooth_l1_loss(pred_dict['traj_with_gt'].squeeze(1), gt_dict['y'], reduction='sum')
        # loss += self.lambda2 * reg_loss

        # # compute scoring gt and loss
        # score_gt = F.softmax(-distance_metric(pred_dict['traj'], gt_dict['y'])/self.temper, dim=-1).detach()
        # # score_loss = torch.sum(torch.mul(- torch.log(pred_dict['score']), score_gt)) / batch_size
        # score_loss = F.binary_cross_entropy(pred_dict['score'], score_gt, reduction='sum')
        # loss += self.lambda3 * score_loss

        #loss_dict = {"tar_cls_loss": cls_loss, "tar_offset_loss": offset_loss, "traj_loss": reg_loss, "score_loss": score_loss}
        loss_dict = {"tar_cls_loss": cls_loss, "tar_offset_loss": offset_loss}
        if self.aux_loss:
            if not isinstance(aux_pred, torch.Tensor) or not isinstance(aux_gt, torch.Tensor):
                return loss, loss_dict
            assert aux_pred.size() == aux_gt.size(), "[TNTLoss]: The dim of prediction and ground truth don't match!"
            aux_loss = F.smooth_l1_loss(aux_pred, aux_gt, reduction="sum")
            # loss += aux_loss / (1.0 if self.reduction == "sum" else batch_size)
            # loss += aux_loss / batch_size
            loss += aux_loss

        return loss, loss_dict
