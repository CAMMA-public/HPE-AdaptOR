# '''
# Project: AdaptOR
# -----
# Copyright (c) University of Strasbourg, All Rights Reserved.
# '''
# modified from https://github.com/rwightman/pytorch-image-models/blob/a2727c1bf78ba0d7b5727f5f95e37fb7f8866b1f/timm/models/layers/split_batchnorm.py

import torch
import torch.nn as nn

__all__ = ["convert_splitgn_model"]

class SplitGroupNorm2d(torch.nn.GroupNorm):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        num_splits: int = 2,
        inference_type="main",
    ) -> None:
        super().__init__(num_groups, num_channels, eps, affine)
        self.inference_type = inference_type
        assert (
            num_splits > 1
        ), "Should have at least one aux GN layer (num_splits at least 2)"
        self.num_splits = num_splits
        self.aux_bn = nn.ModuleList(
            [
                torch.nn.GroupNorm(num_groups, num_channels, eps, affine)
                for _ in range(num_splits - 1)
            ]
        )
    def forward(self, input: torch.Tensor):
        if self.training:
            split_size = input.shape[0] // self.num_splits
            assert (
                input.shape[0] == split_size * self.num_splits
            ), "batch size must be evenly divisible by num_splits: shapes {},{},{}".format(
                input.shape[0], split_size, self.num_splits
            )
            split_input = input.split(split_size)
            x = [super().forward(split_input[0])]
            for i, a in enumerate(self.aux_bn):
                x.append(a(split_input[i + 1]))
            return torch.cat(x, dim=0)
        else:
            if self.inference_type == "main":
                return super().forward(input)
            else:
                split_size = input.shape[0] // (self.num_splits - 1)
                split_input = input.split(split_size)
                x = [m(split_input[i]) for i, m in enumerate(self.aux_bn)]
                return torch.cat(x, dim=0)


def convert_splitgn_model(module, num_splits=2, inference_type="main"):
    mod = module
    if isinstance(module, torch.nn.modules.instancenorm._InstanceNorm):
        return module
    if isinstance(module, torch.nn.modules.normalization.GroupNorm):
        mod = SplitGroupNorm2d(
            module.num_groups,
            module.num_channels,
            module.eps,
            module.affine,
            num_splits=num_splits,
            inference_type=inference_type,
        )
        if module.affine:
            mod.weight.data = module.weight.data.clone().detach()
            mod.bias.data = module.bias.data.clone().detach()
        for aux in mod.aux_bn:
            if module.affine:
                aux.weight.data = module.weight.data.clone().detach()
                aux.bias.data = module.bias.data.clone().detach()
    for name, child in module.named_children():
        if "roi_heads" in name:
            continue
        mod.add_module(name, convert_splitgn_model(child, num_splits=num_splits))
    del module
    return mod
