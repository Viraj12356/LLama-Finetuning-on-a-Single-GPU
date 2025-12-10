#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from typing import Optional, List



class LoRALayer():
    def __init__(
        self, 
        r: int, 
        lora_alpha: int, 
        lora_dropout: float,
        
    ):
        self.r = r
        self.lora_alpha = lora_alpha
        # Optional dropout
        if lora_dropout > 0.:
            self.lora_dropout = nn.Dropout(p=lora_dropout)
        else:
            self.lora_dropout = lambda x: x
        




class Linear(nn.Linear, LoRALayer):
    # LoRA implemented in a dense layer
    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        r: int = 0, 
        lora_alpha: int = 1, 
        lora_dropout: float = 0.,
    ):
        nn.Linear.__init__(self, in_features, out_features)
        LoRALayer.__init__(self, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout,)

        
        # Actual trainable parameters
        if r > 0:
            self.lora_A = nn.Parameter(self.weight.new_zeros((r, in_features)))
            self.lora_B = nn.Parameter(self.weight.new_zeros((out_features, r)))
            self.scaling = self.lora_alpha / self.r
            # Freezing the pre-trained weight matrix
            self.weight.requires_grad = False
        self.reset_parameters()

    def reset_parameters(self):
        nn.Linear.reset_parameters(self)
        if hasattr(self, 'lora_A'):
            # initialize A the same way as the default for nn.Linear and B to zero
            nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
            nn.init.zeros_(self.lora_B)

    def train(self, mode: bool = True):
        
        nn.Linear.train(self) # W + delta_W during training
        if self.r > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
                    

    def forward(self, x: torch.Tensor):

        # h = W.x + delta_W.x = W.x + 
        # W => [in_features x out_features]
        #lora_A - r x in_features
        #lora_B - out_features x r
        #(lora_A * lora_B) = Transpose(r x in_features) * Transpose(out_features x r) = [in_features x r] * [r x out_features] => [in_features x out_features]

        if self.r > 0:
            result = F.linear(x, self.weight, bias=self.bias)  #result = W.x
            result += (self.lora_dropout(x) @ self.lora_A.transpose(0, 1) @ self.lora_B.transpose(0, 1)) * self.scaling # result = result + (lora_A * lora_B).dropout(x)
            return result
        else:
            print("----- R > 0 NOT SATISFIED; PERFORMING NORMAL LINEAR LAYER ---")
            return F.linear(x, self.weight, bias=self.bias)



    







