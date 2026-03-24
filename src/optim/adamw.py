import torch

def build_adamw(model, cfg):
    trainable = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        trainable,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        betas=(0.9, 0.999),
        eps=1e-8,
    )