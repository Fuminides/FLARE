"""Learned category spaces for categorical variables with many categories.

Every training category a of a column gets a point z_a in R^K. An auto-encoder shapes the space: a decoder with one
hidden layer reconstructs the category from its point (cross-entropy weighted by the share pi_a of the training rows
of the category), and a radius loss pulls ||z_a|| towards a target that grows as 1 / sqrt(n_a), mapped linearly onto
[r_min, r_max], so rare categories sit far from the centre, close to each other. The encoder is the point table itself
(an MLP on a one-hot is a lookup). Categories unseen in training use one extra point held at r_max.

The L labels of the variable are Gaussian regions of this space, shared by all the rules (centre mu_l, width sigma_l).
A row of category a belongs to region l to the degree exp(-||z_a - mu_l||^2 / (2 sigma_l^2)). The printed group of a
region is the set of categories whose membership is at least 0.5, so the printed rule is what the model computes.

The first axis of the space starts at the effect of each category on the class (`effects`, from the training rows) and
the warm-up leaves it there. The space is trained alone for a number of steps (`warm_up`) before the model is built,
and then together with the rules, weighting its loss terms with `reconstruction_weight` and `radius_weight`.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class CategorySpace(nn.Module):
    def __init__(self, counts, n_regions, dim=10, hidden=64, r_min=0.5, r_max=3.0, seed=0, sample=1024, effects=None,
                 reconstruction_weight=1.0, radius_weight=1.0):
        super().__init__()
        counts = np.asarray(counts, dtype=np.float64)
        self.n_categories, self.n_regions, self.sample = len(counts), int(n_regions), int(sample)
        inv = 1.0 / np.sqrt(np.maximum(counts, 1.0))
        span = inv.max() - inv.min()
        target = r_min + (r_max - r_min) * ((inv - inv.min()) / span if span > 0 else np.zeros_like(inv))
        self.register_buffer('counts', torch.as_tensor(counts, dtype=torch.float32))
        self.register_buffer('target_radius', torch.as_tensor(np.append(target, r_max), dtype=torch.float32))
        g = torch.Generator().manual_seed(int(seed))
        self.points = nn.Parameter(0.1 * torch.randn(self.n_categories + 1, dim, generator=g))      # last row: unseen
        self.decoder = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, self.n_categories))
        self.centers = nn.Parameter(torch.zeros(self.n_regions, dim))
        self.log_width = nn.Parameter(torch.zeros(self.n_regions))
        self.r_max = float(r_max)
        self.reconstruction_weight = float(reconstruction_weight)
        self.radius_weight = float(radius_weight)
        if self.reconstruction_weight < 0 or self.radius_weight < 0:
            raise ValueError('Category-space loss weights must be nonnegative')
        self.anchored = effects is not None
        if self.anchored:
            e = np.asarray(effects, dtype=np.float64)
            w = counts / counts.sum()
            mean = float((w * e).sum())
            sd = float(np.sqrt((w * (e - mean) ** 2).sum())) or 1.0
            with torch.no_grad():
                self.points[:-1, 0] = torch.as_tensor((e - mean) / sd, dtype=torch.float32)
                self.points[-1, 0] = 0.0                                       # unseen: the prior's place on the axis

    def index(self, x):
        """Category index of each input value: training categories 0..K-1 (frequency ranks), anything else unseen."""
        i = torch.round(x).long()
        return torch.where((i < 0) | (i >= self.n_categories), torch.full_like(i, self.n_categories), i)

    def regions(self, i):
        """[N, L] memberships of category indices i in the L regions."""
        z = self.points[i]
        d2 = (z[:, None, :] - self.centers[None, :, :]).pow(2).sum(dim=2)
        return torch.exp(-d2 / (2.0 * torch.exp(2.0 * self.log_width)[None, :]))

    def forward(self, x):
        return self.regions(self.index(x))

    def loss(self):
        """Reconstruction (on a count-weighted sample of categories) plus the radius term (count-weighted, plus the
        unseen point held at r_max). No labels are used. The sample is drawn from the global random generator."""
        weights = self.counts / self.counts.sum()
        k = min(self.sample, self.n_categories)
        rows = torch.multinomial(weights, k, replacement=True)
        reconstruction = F.cross_entropy(self.decoder(self.points[rows]), rows)
        radius = self.points.norm(dim=1) - self.target_radius
        radius_loss = (weights * radius[:-1].pow(2)).sum() + radius[-1].pow(2) / (self.n_categories + 1)
        return self.reconstruction_weight * reconstruction + self.radius_weight * radius_loss

    def warm_up(self, steps=300, lr=0.01, kmeans_iterations=25):
        """Train the space alone on its loss, then place the regions by count-weighted k-means of the points."""
        optimizer = torch.optim.Adam(list(self.decoder.parameters()) + [self.points], lr=lr)
        axis = self.points[:, 0].detach().clone() if self.anchored else None
        for _ in range(steps):
            optimizer.zero_grad()
            self.loss().backward()
            optimizer.step()
            if axis is not None:                       # the effect axis stays put during the warm-up
                with torch.no_grad():
                    self.points[:, 0] = axis
        with torch.no_grad():
            Z, w = self.points[:-1], self.counts
            L = min(self.n_regions, self.n_categories)
            centers = Z[torch.argsort(w, descending=True)[:L]].clone()             # start at the most frequent categories
            for _ in range(kmeans_iterations):
                assign = torch.cdist(Z, centers).argmin(dim=1)
                for l in range(L):
                    member = assign == l
                    if member.any():
                        centers[l] = (w[member, None] * Z[member]).sum(0) / w[member].sum()
            assign = torch.cdist(Z, centers).argmin(dim=1)
            spread = torch.stack([(Z[assign == l] - centers[l]).norm(dim=1).mean() if (assign == l).any() else torch.tensor(1.0)
                                  for l in range(L)])
            self.centers[:L] = centers
            if L < self.n_regions:                                                   # fewer categories than regions
                self.centers[L:] = centers[torch.arange(self.n_regions - L) % L]
            self.log_width[:L] = torch.log(spread.clamp(min=1e-2) + 1e-2)
            self.log_width[L:] = self.log_width[:L].mean()
        return self

    @torch.no_grad()
    def groups(self, cover=None, threshold=0.5):
        """For every condition (region, or union of regions under a label-run cover [L, R]), the category indices whose
        membership is at least `threshold`, most frequent first."""
        m = self.regions(torch.arange(self.n_categories, device=self.points.device))
        if cover is not None:
            m = torch.clamp(m @ cover.to(m.dtype), max=1.0)
        order = torch.argsort(self.counts, descending=True)
        return [order[(m[order, k] >= threshold)].cpu().numpy() for k in range(m.shape[1])]
