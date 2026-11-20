"""The concept detectors: Inception v3 networks, pretrained on ImageNet, fine-tuned to predict the concepts of an image.

There are two recipes, one for each family of datasets.

`cub` and `sun` follow the concept bottleneck code of Koh et al. (2020): the images are resized to 299 x 299 with no
augmentation and normalised with mean 0.5 and standard deviation 2; a linear output per concept on the 2048 pooled
features; unweighted binary cross-entropy on all the concepts, Adam with learning rate 1e-4, batches of 32, and the
weights of the last epoch (100 epochs for CUB, 30 for SUN). Batch normalisation stays in training mode and dropout
(p = 0.5) acts before the output, as in their code. The network is torchvision's Inception v3 with `transform_input`, which expects the ImageNet
statistics and converts them back.

`derm7pt` (a few hundred images, rare concepts): the same backbone with its auxiliary head, ImageNet normalisation and
augmentation (random resized crops, flips, rotations, mild colour jitter: dermoscopy has no canonical orientation),
binary cross-entropy weighted by each concept's negative / positive ratio plus 0.4 times the auxiliary head's loss, Adam
(learning rate 1e-4, weight decay 1e-4), batches of 16, up to 40 epochs and early stopping on the mean concept AUROC of
the validation images (patience 8, best weights kept). The images are decoded once and kept in memory at 330 pixels.

`predict` returns the sigmoid of the logits, the truth degree of each concept.
"""
import random
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from PIL import Image


@dataclass(frozen=True)
class Recipe:
    kind: str                       # 'koh' or 'derm'
    epochs: int
    batch_size: int
    lr: float = 1e-4
    weight_decay: float = 0.0
    patience: int = 0               # epochs without a better validation AUROC before stopping (derm only)


RECIPES = {'cub': Recipe('koh', 100, 32), 'sun': Recipe('koh', 30, 32), 'derm7pt': Recipe('derm', 40, 16, 1e-4, 1e-4, 8)}


@dataclass
class Rows:
    """The images of a split (paths, or decoded images for the in-memory recipe) and their concepts."""
    paths: list
    C: np.ndarray
    images: list = None

    def subset(self, index):
        return Rows([self.paths[i] for i in index], self.C[index], None if self.images is None else [self.images[i] for i in index])


def load_images(paths, size=330):
    return [Image.open(path).convert('RGB').resize((size, size), Image.BILINEAR) for path in paths]


# --- the networks -------------------------------------------------------------------------------------------------------

def _inception(pretrained, aux_logits):
    from torchvision.models import Inception_V3_Weights, inception_v3
    net = inception_v3(weights=None, aux_logits=aux_logits, transform_input=True, init_weights=False)
    if pretrained:
        state = Inception_V3_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
        net.load_state_dict({k: v for k, v in state.items() if aux_logits or not k.startswith('AuxLogits.')}, strict=True)
    return net


class ConceptInception(nn.Module):
    """Inception v3 up to its pooled 2048 features and one linear output per concept, [N, 3, 299, 299] -> [N, K] logits."""

    def __init__(self, n_concepts, pretrained=True):
        super().__init__()
        self.backbone = _inception(pretrained, aux_logits=False)
        self.backbone.fc = nn.Identity()
        self.concepts = nn.Linear(2048, n_concepts)
        # the new outputs start from a truncated normal (std 0.1, cut at two standard deviations), as in Koh et al.'s code
        nn.init.trunc_normal_(self.concepts.weight, mean=0.0, std=0.1, a=-0.2, b=0.2)

    def forward(self, x):
        return self.concepts(self.backbone(x))


def image_to_tensor(image, size=299):
    """Resize and normalise with mean 0.5 and std 2.0 (Koh et al.'s transform)."""
    array = np.asarray(image.resize((size, size)), dtype=np.float32)
    return (torch.from_numpy(array).permute(2, 0, 1) / 255.0 - 0.5) / 2.0


class ImageFiles(torch.utils.data.Dataset):
    def __init__(self, paths, C):
        self.paths, self.C = paths, torch.as_tensor(C, dtype=torch.float32)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return image_to_tensor(Image.open(self.paths[i]).convert('RGB')), self.C[i]


def build_derm(n_concepts, pretrained=True):
    """Inception v3 with the auxiliary head, both outputs replaced by `n_concepts` logits."""
    net = _inception(pretrained, aux_logits=True)
    net.fc = nn.Linear(net.fc.in_features, n_concepts)
    net.AuxLogits.fc = nn.Linear(net.AuxLogits.fc.in_features, n_concepts)
    return net


def derm_transforms():
    from torchvision import transforms as T
    normalise = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train = T.Compose([T.RandomResizedCrop(299, scale=(0.7, 1.0)), T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
                       T.RandomRotation(30), T.ColorJitter(0.1, 0.1, 0.1, 0.02), T.ToTensor(), normalise])
    test = T.Compose([T.CenterCrop(299), T.ToTensor(), normalise])
    return train, test


class Decoded(torch.utils.data.Dataset):
    def __init__(self, images, C, transform):
        self.images, self.C, self.transform = images, torch.as_tensor(C, dtype=torch.float32), transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        return self.transform(self.images[i]), self.C[i]


def mean_auroc(C, P):
    """Mean AUROC of the probabilities P against the 0/1 concepts C, over the concepts that have both values."""
    from sklearn.metrics import roc_auc_score
    return float(np.mean([roc_auc_score(C[:, j], P[:, j]) for j in range(C.shape[1]) if 0 < C[:, j].sum() < len(C)]))


# --- training and prediction --------------------------------------------------------------------------------------------

def _seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True       # convolution autotuning would make runs differ
        torch.backends.cudnn.benchmark = False


def _loader(dataset, batch_size, shuffle, seed, workers, device, drop_last=False):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last,
                                       generator=torch.Generator().manual_seed(seed), num_workers=workers,
                                       persistent_workers=workers > 0, pin_memory=torch.device(device).type == 'cuda')


def fit_detector(recipe, train, val, seed, device='cpu', workers=0, epochs=None, batch_size=None, pretrained=True, log=print):
    """Train a detector on `train` (Rows) and return it. `val` (Rows) is used by the recipe that stops early; `epochs` and
    `batch_size` override the recipe's (for tests and smoke runs)."""
    epochs, batch_size = epochs or recipe.epochs, batch_size or recipe.batch_size
    _seed(seed)
    n_concepts = train.C.shape[1]
    if recipe.kind == 'koh':
        model = ConceptInception(n_concepts, pretrained).to(device)
        loader = _loader(ImageFiles(train.paths, train.C), batch_size, True, seed, workers, device)
        optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=recipe.lr)
        criterion = nn.BCEWithLogitsLoss()
        model.train()
        for epoch in range(epochs):
            losses = []
            for x, c in loader:
                loss = criterion(model(x.to(device)), c.to(device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.detach()))
            log(f'seed={seed} epoch={epoch + 1} loss={np.mean(losses):.5f}')
        return model

    train_tf, test_tf = derm_transforms()
    positives = train.C.sum(axis=0)
    pos_weight = torch.as_tensor((len(train.C) - positives) / np.maximum(positives, 1.0), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    model = build_derm(n_concepts, pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=recipe.lr, weight_decay=recipe.weight_decay)
    loader = _loader(Decoded(train.images, train.C, train_tf), batch_size, True, seed, workers, device, drop_last=True)
    best, best_state, waited = -np.inf, None, 0
    for epoch in range(epochs):
        model.train()
        for x, c in loader:
            x, c = x.to(device), c.to(device)
            logits, aux = model(x)
            loss = criterion(logits, c) + 0.4 * criterion(aux, c)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        score = mean_auroc(val.C, predict(recipe, model, val, device, workers=workers))
        log(f'seed={seed} epoch={epoch + 1} validation_auroc={score:.4f}')
        if score > best:
            best, waited = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            waited += 1
            if waited >= recipe.patience:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def predict(recipe, model, rows, device='cpu', batch_size=64, workers=0):
    """The sigmoid of the detector's logits on `rows`: [N, K] truth degrees."""
    model.eval()
    if recipe.kind == 'koh':
        dataset = ImageFiles(rows.paths, rows.C)
    else:
        dataset = Decoded(rows.images, rows.C, derm_transforms()[1])
    loader = _loader(dataset, batch_size, False, 0, workers, device)
    return torch.cat([torch.sigmoid(model(x.to(device))).cpu() for x, _ in loader]).numpy().astype(np.float32)


def save_detector(path, model, recipe_name, concept_names, seed):
    torch.save({'state_dict': model.state_dict(), 'recipe': recipe_name, 'concept_names': concept_names, 'seed': seed}, path)


def load_detector(path, device='cpu'):
    """(model, recipe, concept names) of a checkpoint written by save_detector."""
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    recipe = RECIPES[checkpoint['recipe']]
    names = checkpoint['concept_names']
    model = (ConceptInception(len(names), pretrained=False) if recipe.kind == 'koh' else build_derm(len(names), pretrained=False))
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device), recipe, names
