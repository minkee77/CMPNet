import torch
import torch.nn as nn
import torch.nn.functional as F

import models
import utils
from .models import register

@register('meta-baseline')
class MetaBaseline(nn.Module):

    def __init__(self, encoder, encoder_args={}, method='sqr',
                 temp=1., temp_learnable=False, progressive= True):
        super().__init__()
        self.encoder = models.make(encoder, **encoder_args)
        self.method = method
        self.progressive = progressive

        if temp_learnable:
            self.temp = nn.Parameter(torch.tensor(temp))
        else:
            self.temp = temp

    def forward_label(self, x_shot, x_query):
        shot_shape = x_shot.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_query], dim=0))
        x_shot, x_query = x_tot[:len(x_shot)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot = x_shot.mean(dim=-2)
            x_shot = F.normalize(x_shot, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot = x_shot.mean(dim=-2)
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, x_shot, metric=metric, temp=self.temp)
        return logits
    
    def forward_unlabel(self, x_shot, x_unlabel, x_query):
        shot_shape = x_shot.shape[:-3]
        unlabel_shape = x_unlabel.shape[:-3]
        query_shape = x_query.shape[:-3]
        img_shape = x_shot.shape[-3:]

        x_shot = x_shot.view(-1, *img_shape)
        x_unlabel = x_unlabel.view(-1, *img_shape)
        x_query = x_query.view(-1, *img_shape)
        x_tot = self.encoder(torch.cat([x_shot, x_unlabel, x_query], dim=0))
        x_shot, x_unlabel, x_query = x_tot[:len(x_shot)], x_tot[len(x_shot):len(x_shot)+len(x_unlabel)], x_tot[-len(x_query):]
        x_shot = x_shot.view(*shot_shape, -1)
        x_unlabel = x_unlabel.view(*unlabel_shape, -1)
        x_query = x_query.view(*query_shape, -1)

        if self.method == 'cos':
            x_shot_tmp = x_shot.mean(dim=-2)
            x_shot_tmp = F.normalize(x_shot_tmp, dim=-1)
            x_unlabel_tmp = F.normalize(x_unlabel, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            x_shot_tmp = x_shot.mean(dim=-2)
            x_unlabel_tmp = x_unlabel
            metric = 'sqr'

        logits = utils.compute_logits(
                x_unlabel_tmp, x_shot_tmp, metric=metric, temp=self.temp)

        prob_unlabel = F.softmax(logits, 2)

        B, n_way, n_shot = shot_shape
        prob_shot = torch.arange(n_way).unsqueeze(1).expand(n_way, n_shot).reshape(-1)
        prob_shot = torch.zeros(n_way*n_shot, n_way).scatter_(1, prob_shot.unsqueeze(1), 1).repeat(B, 1, 1)
        prob_shot = prob_shot.cuda()
        x_shot = x_shot.view(B, n_way*n_shot, -1)

        x_all = torch.cat((x_shot, x_unlabel),1)
        prob_all = torch.cat((prob_shot, prob_unlabel), 1)

        prob_sum = torch.sum(prob_all, dim=1, keepdim=True)
        prob = prob_all/prob_sum
        cluster_center = torch.sum(x_all.unsqueeze(2)*prob.unsqueeze(3), dim=1)

        if self.method == 'cos':
            cluster_center = F.normalize(cluster_center, dim=-1)
            x_query = F.normalize(x_query, dim=-1)
            metric = 'dot'
        elif self.method == 'sqr':
            metric = 'sqr'

        logits = utils.compute_logits(
                x_query, cluster_center, metric=metric, temp=self.temp)
        return logits

    def forward(self, *input):
        if not self.progressive:
            x_shot, x_query = input
            logits = self.forward_label(x_shot, x_query)
        else:
            x_shot, x_unlabel, x_query = input
            logits = self.forward_unlabel(x_shot, x_unlabel, x_query)

        return logits




