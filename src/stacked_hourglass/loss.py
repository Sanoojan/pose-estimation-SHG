import torch.nn as nn
from torch.nn.functional import mse_loss, kl_div, softmax
import torch

def joints_mse_loss(output, target, target_weight=None):
    batch_size = output.size(0)
    num_joints = output.size(1)
    heatmaps_pred = output.view((batch_size, num_joints, -1)).split(1, 1)
    heatmaps_gt = target.view((batch_size, num_joints, -1)).split(1, 1)

    loss = 0
    for idx in range(num_joints):
        heatmap_pred = heatmaps_pred[idx]
        heatmap_gt = heatmaps_gt[idx]
        if target_weight is None:
            loss += 0.5 * mse_loss(heatmap_pred, heatmap_gt, reduction='mean')
        else:
            loss += 0.5 * mse_loss(
                heatmap_pred.mul(target_weight[:, idx]),
                heatmap_gt.mul(target_weight[:, idx]),
                reduction='mean'
            )

    return loss / num_joints


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super().__init__()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        if not self.use_target_weight:
            target_weight = None
        return joints_mse_loss(output, target, target_weight)


def kldiv_distill_loss(output,return_all_block_outs=None,final_output=None):
    '''Loss fuction for self Distillation
    output: 2dim List of Tensors with 2nd axis 1 TODO:JGB: find why?
    '''
    batch_size = output[1][0].size(0)
    loss = 0
    t=3.0
    # for HG Latent in Middle
    # last = torch.clamp(softmax((output[-1][0]), dim=1)/t,1e-7,1)
    # for i in range(len(output)-1):
    #     curr = softmax(output[i][0], dim=1)
    #     curr=torch.clamp(curr/t, 1e-7, 1)
    #     loss+=kl_div(curr.log(), last, reduction='mean')

    ## for Feature maps
    
    # last = torch.clamp(softmax((output[-1][1]), dim=1)/t,1e-7,1)
    # for i in range(len(output)-1):
    #     curr = softmax(output[i][1], dim=1)
    #     curr=torch.clamp(curr/t, 1e-7, 1)
    #     loss+=kl_div(curr.log(), last, reduction='mean')


    # for all res blocks
    last_hg = return_all_block_outs[-1]
    for i in range(len(return_all_block_outs)-1):
        for j in range(len(return_all_block_outs[i])):
            curr = softmax(return_all_block_outs[i][j], dim=1)
            curr=torch.clamp(curr/t, 1e-7, 1)
            last = softmax(last_hg[j], dim=1)
            last=torch.clamp(last/t, 1e-7, 1)
            loss+=kl_div(curr.log(), last, reduction='mean')
    
    # last_hg = return_all_block_outs[-1]
    # for i in range(len(return_all_block_outs)-1):
    #     j=0
    #     curr = softmax(return_all_block_outs[i][j], dim=1)
    #     curr=torch.clamp(curr, 1e-7, 1)
    #     last = softmax(last_hg[j], dim=1)
    #     last=torch.clamp(last, 1e-7, 1)
    #     loss+=kl_div(curr.log(), last, reduction='mean')
         


    # print(loss)
    return loss