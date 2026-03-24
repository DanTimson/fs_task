from .muon import split_muon_params, Muon
import torch


def build_hybrid(model, cfg):
    muon_params, adam_params = split_muon_params(model)

    muon_optim = Muon(
        muon_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    adam_optim = torch.optim.AdamW(
        adam_params,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )

    return muon_optim, adam_optim