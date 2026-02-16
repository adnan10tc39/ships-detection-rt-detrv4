"""
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
"""

import copy
from typing import Tuple
import torch.nn as nn

try:
    from calflops import calculate_flops
except Exception:
    calculate_flops = None


class _OpenVocabWrapper(nn.Module):
    def __init__(self, model, text_prompts=None, text_embeddings=None):
        super().__init__()
        self.model = model
        self.text_prompts = text_prompts
        self.text_embeddings = text_embeddings

    def forward(self, images):
        return self.model(
            images,
            text_prompts=self.text_prompts,
            text_embeddings=self.text_embeddings,
        )


def _get_open_vocab_prompts(cfg):
    text_prompts = None
    try:
        text_prompts = getattr(cfg.train_dataloader.dataset, 'text_prompts', None)
    except Exception:
        text_prompts = None
    if text_prompts is None:
        text_prompts = cfg.yaml_cfg.get('DFINETransformer', {}).get('text_prompts')
    return text_prompts

def stats(
    cfg,
    input_shape: Tuple=(1, 3, 640, 640), ) -> Tuple[int, dict]:

    base_size = cfg.train_dataloader.collate_fn.base_size
    input_shape = (1, 3, base_size, base_size)

    model_for_info = copy.deepcopy(cfg.model).deploy()
    # Set to training mode to use dynamic position embeddings during profiling
    model_for_info.train()
    # Ensure encoder uses dynamic positional embeddings by clearing eval_spatial_size
    if hasattr(model_for_info, 'encoder') and hasattr(model_for_info.encoder, 'eval_spatial_size'):
        model_for_info.encoder.eval_spatial_size = None
    decoder = getattr(model_for_info, 'decoder', None)
    if decoder is not None and getattr(decoder, 'open_vocab', False):
        text_prompts = _get_open_vocab_prompts(cfg)
        if text_prompts is None:
            params = sum(p.numel() for p in model_for_info.parameters())
            del model_for_info
            return params, {"Model FLOPs: skipped (open-vocab prompts missing)  Params:%s" % (params,)}
        model_for_info = _OpenVocabWrapper(model_for_info, text_prompts=text_prompts)

    params = sum(p.numel() for p in model_for_info.parameters())
    if calculate_flops is None:
        del model_for_info
        return params, {"Model FLOPs: skipped (calflops not installed)   Params:%s" % (params,)}

    flops, macs, _ = calculate_flops(
        model=model_for_info,
        input_shape=input_shape,
        output_as_string=True,
        output_precision=4,
        print_detailed=False,
    )
    del model_for_info

    return params, {"Model FLOPs:%s   MACs:%s   Params:%s" % (flops, macs, params)}
