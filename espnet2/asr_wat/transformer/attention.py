#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Multi-Head Attention layer definition."""

import math
import random

import torch
from torch import nn


class AdpHeadedAttention(nn.Module):

    def __init__(self, n_head, n_feat, n_attn, dropout_rate, threshold):
        """Construct an MultiHeadedAttention object."""
        #n_feat dim after pre-encoder (e.g. conv2d)
        #n_attn total dim of each attention layer 
        super(AdpHeadedAttention, self).__init__()
        assert n_attn % n_head == 0
        self.d_k = n_attn // n_head
        self.h = n_head
        self.linear_q = nn.Linear(n_feat, n_attn)
        self.linear_k = nn.Linear(n_feat, n_attn)
        self.linear_v = nn.Linear(n_feat, n_attn)
        self.linear_out = nn.Linear(n_attn, n_feat)

        self.attn = None        
        self.dropout = nn.Dropout(p=dropout_rate)       
        self.threshold = threshold
        self.scale_a = torch.nn.Parameter(torch.FloatTensor(torch.ones(n_attn)))
        self.scale_base = torch.ones(n_attn)
        self.scale_max = None
        
    def forward_qkv(self, query, key, value):
        n_batch = query.size(0) 

        norm_q = torch.linalg.norm(self.linear_q.weight, ord=2, dim=1)+(random.random()*(1e-10))
        norm_k = torch.linalg.norm(self.linear_k.weight, ord=2, dim=1)+(random.random()*(1e-10))

        v = self.linear_v(value).view(n_batch, -1, self.h, self.d_k)       
        q = self.linear_q(query).view(n_batch, -1, self.h * self.d_k)
        k = self.linear_k(key).view(n_batch, -1, self.h * self.d_k)
        
        self.scale_max = torch.linalg.norm(self.scale_a, ord=float('inf')).to(self.scale_a.device)
        scale_mask = torch.gt(torch.abs(self.scale_a), self.scale_base.to(self.scale_a.device) * (self.threshold * self.scale_max)).float()
        scale_wat = self.scale_a * scale_mask
        
            
        q = q * torch.unsqueeze(torch.unsqueeze(scale_wat,0),0)
        
        q = q/(torch.unsqueeze(torch.unsqueeze(norm_q,0),0))
        k = k/(torch.unsqueeze(torch.unsqueeze(norm_k,0),0))
        
        
        k =k.view(n_batch, -1, self.h, self.d_k)
        q =q.view(n_batch, -1, self.h, self.d_k)

        norm_wat = torch.linalg.norm(scale_wat, ord=1, dim=0)
        
        q = q.transpose(1, 2)  # (batch, head, time1, d_k)
        k = k.transpose(1, 2)  # (batch, head, time2, d_k)
        v = v.transpose(1, 2)  # (batch, head, time2, d_k)

        return q, k, v, norm_wat, scale_wat

    def forward_attention(self, value, scores, mask):
        """Compute attention context vector.

        Args:
            value (torch.Tensor): Transformed value (#batch, n_head, time2, d_k).
            scores (torch.Tensor): Attention score (#batch, n_head, time1, time2).
            mask (torch.Tensor): Mask (#batch, 1, time2) or (#batch, time1, time2).

        Returns:
            torch.Tensor: Transformed value (#batch, time1, d_model)
                weighted by the attention score (#batch, time1, time2).

        """
        n_batch = value.size(0)
        if mask is not None:
            mask = mask.unsqueeze(1).eq(0)  # (batch, 1, *, time2)
            min_value = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(mask, min_value)
            self.attn = torch.softmax(scores, dim=-1).masked_fill(
                mask, 0.0
            )  # (batch, head, time1, time2)
        else:
            self.attn = torch.softmax(scores, dim=-1)  # (batch, head, time1, time2)

        p_attn = self.dropout(self.attn)
        x = torch.matmul(p_attn, value)  # (batch, head, time1, d_k)
        x = (
            x.transpose(1, 2).contiguous().view(n_batch, -1, self.h * self.d_k)
        )  # (batch, time1, d_model)

        return self.linear_out(x)  # (batch, time1, d_model)

    def forward(self, query, key, value, mask):
        """Compute scaled dot product attention.

        Args:
            query (torch.Tensor): Query tensor (#batch, time1, size).
            key (torch.Tensor): Key tensor (#batch, time2, size).
            value (torch.Tensor): Value tensor (#batch, time2, size).
            mask (torch.Tensor): Mask tensor (#batch, 1, time2) or
                (#batch, time1, time2).

        Returns:
            torch.Tensor: Output tensor (#batch, time1, d_model).

        """
        q, k, v, norm_wat, scale_wat = self.forward_qkv(query, key, value)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        return self.forward_attention(v, scores, mask), norm_wat, scale_wat

