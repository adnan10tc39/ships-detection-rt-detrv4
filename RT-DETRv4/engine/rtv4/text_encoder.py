"""
Text encoder utilities for open-vocabulary RT-DETR.
"""

from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import CLIPTokenizer, CLIPTextModel

from ..core import register

__all__ = ['CLIPTextEncoder']


def _format_prompt(template: str, class_name: str) -> str:
    if '{class}' in template:
        return template.format(**{"class": class_name})
    if '{}' in template:
        return template.format(class_name)
    return f"{template} {class_name}".strip()


@register()
class CLIPTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        cache_dir: Optional[str] = None,
        max_length: int = 77,
        normalize: bool = True,
        freeze: bool = True,
        cache_text: bool = True,
        cache_on_cpu: bool = True,
    ):
        super().__init__()
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.normalize = normalize
        self.freeze = freeze
        self.cache_text = cache_text
        self.cache_on_cpu = cache_on_cpu

        self.tokenizer = CLIPTokenizer.from_pretrained(model_name, cache_dir=cache_dir)
        self.text_model = CLIPTextModel.from_pretrained(model_name, cache_dir=cache_dir)

        if self.freeze:
            self.text_model.eval()
            for p in self.text_model.parameters():
                p.requires_grad = False

        self._text_cache = {}

    @torch.no_grad()
    def encode(self, texts: Sequence[str], device=None, dtype=None) -> torch.Tensor:
        if isinstance(texts, str):
            texts = [texts]
        cache_key = tuple(texts)
        if self.cache_text and cache_key in self._text_cache:
            cached = self._text_cache[cache_key]
            if device is not None:
                cached = cached.to(device)
            if dtype is not None:
                cached = cached.to(dtype)
            return cached

        if device is None:
            device = next(self.text_model.parameters()).device
        inputs = self.tokenizer(
            list(texts),
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = self.text_model(**inputs)
        embeddings = outputs.pooler_output
        if self.normalize:
            embeddings = F.normalize(embeddings, dim=-1)
        if dtype is not None:
            embeddings = embeddings.to(dtype)

        if self.cache_text:
            cached = embeddings.detach().cpu() if self.cache_on_cpu else embeddings.detach()
            self._text_cache[cache_key] = cached
        return embeddings

    @torch.no_grad()
    def encode_prompts(
        self,
        class_names: Sequence[str],
        templates: Optional[Sequence[str]] = None,
        device=None,
        dtype=None,
    ) -> torch.Tensor:
        if templates is None:
            templates = ["{class}"]
        if isinstance(templates, str):
            templates = [templates]
        prompts = []
        for name in class_names:
            for template in templates:
                prompts.append(_format_prompt(template, name))
        embeddings = self.encode(prompts, device=device, dtype=dtype)
        if len(templates) > 1:
            embeddings = embeddings.view(len(class_names), len(templates), -1).mean(dim=1)
        return embeddings

    def clear_cache(self):
        self._text_cache.clear()

