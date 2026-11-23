"""Frozen CLIP text encoder providing the sentence and word (per-token) embeddings used by HyperCGAN."""

import clip
import torch

#----------------------------------------------------------------------------

class CLIPTextEncoder:
    """
    Wraps CLIP ViT-B/32. For a batch of CLIP token ids it returns
      - txt_fts: [batch_size, 512]      L2-normalized sentence embeddings (`encode_text`),
      - wrd_fts: [batch_size, 77, 512]  L2-normalized per-token embeddings (output of `ln_final`).
    """
    def __init__(self, device, model_name='ViT-B/32'):
        self.device = torch.device(device)
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval().requires_grad_(False)
        self._ln_final = None
        self.model.ln_final.register_forward_hook(self._hook)

    def _hook(self, module, inputs, output):
        self._ln_final = output.detach()

    @staticmethod
    def tokenize(texts, device='cpu'):
        return clip.tokenize(texts, truncate=True).to(device)

    @torch.no_grad()
    def encode(self, tokens):
        tokens = tokens.to(self.device)
        txt_fts = self.model.encode_text(tokens)
        wrd_fts = self._ln_final
        txt_fts = txt_fts / txt_fts.norm(dim=-1, keepdim=True)
        wrd_fts = wrd_fts / wrd_fts.norm(dim=-1, keepdim=True)
        return txt_fts, wrd_fts

#----------------------------------------------------------------------------
