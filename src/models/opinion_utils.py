"""Helpers that work for both Dirichlet (B,C) and Beta (B,C,2) opinions."""
import numpy as np
import torch


def opinion_prob(op):
    """Return (B, C) predictive probabilities from an opinion dict."""
    if "prob" in op:
        return op["prob"]
    alpha = op["alpha"]
    return alpha / alpha.sum(dim=-1, keepdim=True)


def opinion_uncertainty(op):
    """Return (B,) scalar vacuity."""
    u = op["uncertainty"]
    return u if u.dim() == 1 else u.mean(dim=-1)
