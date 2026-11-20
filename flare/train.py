"""Training of FLARE.

`train_loop` trains one model (or the blocks of a joint model) by gradient descent, and `fit_fold` is the whole recipe
for one fold: partitions, model, the way the averaged models are trained (bagging.py), the evaluation of the deployed
model and the recalibration of its probabilities. In short:

* the loss is the cross-entropy of the class scores, with class weights inversely proportional to the class
  frequencies, plus the reconstruction and radius terms of the category spaces;
* the uniform share k of the gradient grafting decreases from 1 towards 0 with two oscillations,
  k = xi + 0.2 sin(4 pi xi), xi being the share of epochs that remain;
* Adam with four groups of steps: the points and biases (lr), the selectors (lr * lr_structure), the crossings of the
  partitions (that times crossings_lr) and the category spaces (times category_space_lr); every step decays along a
  cosine to 1e-6; the update of the interaction rules is multiplied by lr_interactions after the Adam step; the gradient
  of a model is rescaled when its norm exceeds 1;
* the epoch that is kept is the one with the best accuracy of the discrete model on the validation rows.
"""
import time
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from . import bagging
from .bagging import PARALLEL_BAGS_MAX_ELEMENTS
from .calibration import calibrate_probabilities
from .category_space import CategorySpace
from .cuda_graph import GraphedStep, InteractionShrink
from .data import fold_batch_size, fold_loaders
from .inputs import TabularInputs
from .labels import label_run_list
from .metrics import (balanced_class_weights, eval_chunk_rows, evaluate, probability_metrics, rule_set_complexity,
                      rule_set_size)
from .model import BagEnsemble, FLAREModel
from .partitions import quantile_partitions
from .selectors import GradMode


def k_schedule(epoch, epochs, cyclical=True):
    """The uniform share k of the epoch: xi + 0.2 sin(4 pi xi) with xi = (epochs - epoch) / epochs (or just xi)."""
    xi = (epochs - epoch) / epochs
    return xi + 0.2 * np.sin(2 * np.pi * 2 * xi) if cyclical else xi


@torch.no_grad()
def deciding_rule_mask(model, X):
    """[R] bool: rules that win at least one row of X in the model's current mode."""
    decided = torch.zeros(model.nrules, dtype=torch.bool, device=X.device)
    for chunk in X.split(eval_chunk_rows(model)):
        _, scores = model(chunk)
        decided[scores.max(dim=2)[0].argmax(dim=1)] = True
    return decided


def train_loop(model, train_loader, val_loader, config, class_weights=None, parallel_bags=0, device='cpu', verbose=False):
    """Train `model` and leave in it the parameters of its best validation epoch. Returns (losses, validation
    accuracies) per epoch. parallel_bags = B > 0 trains the B blocks of a joint model as independent models: the
    loaders then carry each row's block memberships next to its label (bagging.parallel_bag_loaders)."""
    epochs = config.epochs
    inputs = model.inputs
    crossings = inputs.crossing_parameters() if inputs is not None else []
    spaces = inputs.space_parameters() if inputs is not None else []

    # Four groups of parameters with their own step size (Adam is invariant to the scale of the gradient, so scaling the
    # loss is not enough: they need real parameter groups). The crossings and the category spaces have step sizes
    # relative to the selectors'. The decision head, the crossings and the spaces are not "structure".
    head = {id(p) for p in model.head.parameters()} | {id(p) for p in crossings} | {id(p) for p in spaces}
    structure = [p for p in model.parameters() if id(p) not in head]
    lr = config.lr
    optimizer = torch.optim.Adam([{'params': structure, 'lr': lr * config.lr_structure},
                                  {'params': list(model.head.parameters()), 'lr': lr}]
                                 + ([{'params': crossings, 'lr': lr * config.lr_structure * config.crossings_lr}] if crossings else [])
                                 + ([{'params': spaces, 'lr': lr * config.lr_structure * config.category_space_lr}] if spaces else []),
                                 lr=lr, weight_decay=config.weight_decay)
    # Slower interaction rules: their rows share tensors with the one-condition rules, so their Adam update is scaled
    # after each step instead (p <- p_old + f * (p_new - p_old)).
    interaction_rows = None
    if config.lr_interactions != 1.0:
        rows = model.interaction_rows()
        if bool(rows.any()):
            interaction_rows = InteractionShrink([(parameter, dim, rows.nonzero().flatten().to(parameter.device))
                                                  for parameter, dim in model.rule_parameters()], config.lr_interactions)

    # Parallel union bagging: the joint model's blocks are independent models, each with its own loss, clipping,
    # validation split and best-validation checkpoint, so the result is the union of separately trained models.
    best_blocks = best_state = None
    if parallel_bags:
        if model.head.joint_bags != parallel_bags:
            raise ValueError('parallel bags need a model built with joint_bags equal to the number of bags')
        bagging.block_parameters(model, parallel_bags)
        best_blocks = np.full(parallel_bags, -np.inf)
        block_since, block_stopped = np.zeros(parallel_bags, dtype=int), np.zeros(parallel_bags, dtype=bool)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    train_mode = model.gradMode
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    k_lower = config.k_lower_train

    def compute_loss(X, y):
        """Forward and loss of one training step. Everything in here is recorded when the CUDA graph is active, so it
        must not need the CPU (no .item(), no branching on tensor values)."""
        if parallel_bags:
            member = y[:, 1:1 + parallel_bags].float()                                   # [B, K] training rows of each block
            y = y[:, 0]
        output, _ = model(X)
        loss = bagging.parallel_bag_loss(model.head.bag_outputs, y, member, class_weights) if parallel_bags else criterion(output, y)
        if spaces:                                       # the category spaces' reconstruction and radius terms
            loss = loss + inputs.space_loss(X.device)
        return loss

    graphed_step = GraphedStep(compute_loss, model.parameters()) \
        if config.cuda_graph and torch.device(device).type == 'cuda' and train_mode == GradMode.GRAFT else None
    rows_seen = train_loader.dataset.X

    best_acc, epochs_no_improve = float('-inf'), 0
    losses, val_acc_history = [], []
    for epoch in range(epochs):
        if train_mode == GradMode.CONCRETE:
            k_lower = 0.1                                # any value bigger than 0: with k = 0 the Gumbel choice is hard
            model.set_temperature(0.01 ** (epoch / epochs))   # the selectors' temperature is annealed from 1 to 0.01
        actual_k = max(min(k_schedule(epoch, epochs, config.cyclical_k), config.k_upper_train), k_lower)
        model.set_train_mode(train_mode, actual_k)
        if graphed_step is not None:
            graphed_step.set_k(actual_k)
        if train_mode == GradMode.CONCRETE:
            # This pass does not change the model, but with Gumbel selectors every forward draws noise from the global
            # generator, so removing it changes all the later draws (and so the results).
            deciding_rule_mask(model, rows_seen.to(device))

        epoch_loss_sum, epoch_batches = 0.0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            # Drop the previous step's loss first: an eager loss keeps its autograd graph alive, and with it gradient
            # accumulators bound to the default stream, which a CUDA graph capture must not depend on.
            loss = None
            loss = graphed_step(X, y) if graphed_step is not None and graphed_step.usable() else None
            if loss is None:                             # eager step (graph off, unsupported setting, or recording failed)
                optimizer.zero_grad()
                loss = compute_loss(X, y)
                loss.backward(retain_graph=False)
            if parallel_bags:
                bagging.clip_grad_norm_blocks(model, parallel_bags, max_norm=1.0)
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            epoch_loss_sum += loss.detach().double()
            epoch_batches += 1
            if interaction_rows is not None:
                interaction_rows.before_step()
            optimizer.step()
            if interaction_rows is not None:
                interaction_rows.after_step()
        losses.append(float(epoch_loss_sum) / max(epoch_batches, 1))

        # Validation of the epoch. The epoch that is kept is chosen with the DISCRETE model (PURE), the one deployed.
        if parallel_bags:
            block_scores = bagging.block_validation(model, val_loader, parallel_bags, config.checkpoint_metric)
            val_acc = val_acc_pure = val_f1_pure = float(block_scores.mean())
        elif train_mode == GradMode.GRAFT:
            # The grafted forward is the PURE forward, so a training-mode validation pass would repeat the PURE one. The
            # loader is still iterated once: it shuffles, and every pass draws from the global generator, so skipping
            # this one changes the later training batches.
            for _ in val_loader:
                pass
            val_acc_pure, val_f1_pure = evaluate(model, val_loader, 0.0, GradMode.PURE)[:2]
            val_acc = val_acc_pure
        else:
            val_acc = evaluate(model, val_loader, 0.0, model.gradMode)[0]        # the Gumbel model itself (draws noise)
            val_acc_pure, val_f1_pure = evaluate(model, val_loader, 0.0, GradMode.PURE)[:2]
        if verbose and epoch % 20 == 0:
            print(f'Epoch {epoch} - loss {losses[-1]:.4f} - validation {val_acc:.4f} - pure {val_acc_pure:.4f} - best {best_acc:.4f}', flush=True)

        if config.checkpoint_metric == 'f1':             # select the epoch on validation macro-F1 instead of accuracy
            val_acc_pure = val_f1_pure
        if parallel_bags:
            # Per-block checkpoint: each block keeps the parameters of its own best validation epoch. With patience, a
            # block whose model alone would have stopped keeps its checkpoint from then on; the run ends when every block
            # has stopped.
            better = (block_scores > best_blocks) & ~block_stopped
            improved = np.flatnonzero(better)
            if best_state is None:
                best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            elif len(improved):
                bagging.copy_blocks(best_state, model, parallel_bags, improved)
            best_blocks[improved] = block_scores[improved]
            block_since[better] = 0
            block_since[~better & ~block_stopped] += 1
            if config.patience > 0:
                block_stopped |= block_since >= config.patience
            epochs_no_improve = config.patience if config.patience > 0 and block_stopped.all() else 0
            best_acc = float(best_blocks.mean())
        elif val_acc_pure > best_acc:
            # Keep the best epoch (a strict improvement: the first epoch that reaches the best score).
            best_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
            best_acc = val_acc_pure
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step()
        val_acc_history.append(val_acc)
        if config.patience > 0 and epochs_no_improve >= config.patience:
            break

    # The model that is deployed is the one of the best validation epoch.
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose and best_blocks is not None:
        print('Parallel bags: best validation per block ' + ', '.join(f'{v:.3f}' for v in best_blocks), flush=True)
    return losses, val_acc_history


@dataclass
class FitResult:
    model: nn.Module
    fold: object
    train_loader: object
    val_loader: object
    test_loader: object
    config: object
    time2train: float = 0.0
    losses: list = field(default_factory=list)
    val_accs: list = field(default_factory=list)
    route: str = ''                 # 'single', 'sequential', 'parallel' or 'staged': how the averaged models were trained


def fit_fold(fold, config, cardinalities, random_seed, device=None, verbose=False, max_parallel_elements=None,
             inputs_factory=None, fixed_order_evaluation=False):
    """Train FLARE on a preprocessed fold (preprocessing.Fold) and return a FitResult.

    cardinalities: number of categories of each categorical column of the fold (in column order; for the whole dataset,
        not only the training rows: it sets the size of the selectors). Columns turned into category spaces by the
        preprocessing are real and do not appear here.
    random_seed: seeds the global generators (initial values, batch order), the category spaces and the resplits of the
        averaged models. The experiments use the seed of the run plus the fold index; the split of the training rows
        into training and validation is already in `fold` (it uses the seed of the run alone).
    max_parallel_elements: overrides PARALLEL_BAGS_MAX_ELEMENTS (bagging.py), the step size above which the averaged
        models are trained one at a time.
    inputs_factory: for experiments in which the memberships of the real variables are given (concept bottleneck
        models, graphs), a function device -> inputs module (the contract is in inputs.py) that replaces the tabular
        partitions. It is called once per model that is built. If the module has trainable parameters, they are
        learned by the first averaged model and shared, frozen, by the others, as the tabular partitions are.
    fixed_order_evaluation: validation and test loaders without shuffling (data.fold_loaders); the concept bottleneck
        experiments use it.
    """
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    max_elements = PARALLEL_BAGS_MAX_ELEMENTS if max_parallel_elements is None else max_parallel_elements
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    n_fold = len(fold.y_train) + len(fold.y_val)
    batch_size = fold_batch_size(n_fold, config.batch_size, config.steps_per_epoch)
    train_loader, val_loader, test_loader = fold_loaders(fold, batch_size, device, fixed_order_evaluation)
    categorical = np.asarray(fold.categorical, dtype=bool)
    real_columns = [ix for ix, c in enumerate(categorical) if not c]
    n_categorical = int(categorical.sum())
    n_classes = fold.n_classes
    mode = GradMode.from_string(config.grad_mode)
    if len(cardinalities) != n_categorical:
        raise ValueError(f'{len(cardinalities)} cardinalities for {n_categorical} categorical columns')

    tac = time.time()
    # Initial partitions: quantiles of the training rows as the model will read them (float32, the real columns of the
    # fold; the columns that are category spaces are real here: they hold frequency ranks).
    partitions = None
    if real_columns and inputs_factory is None:
        partitions = quantile_partitions(np.asarray(train_loader.dataset.X.cpu().numpy())[:, real_columns], config.labels)
    class_weights = balanced_class_weights(fold.y_train, n_classes, device) if config.class_weight == 'balanced' else None
    space_columns = fold.category_space if config.category_space > 0 else None

    def build_spaces():
        """A warmed-up CategorySpace for each category-space column, keyed by its index among the real variables. The
        warm-up runs before the model is built and draws from the global generator, so it fixes how many draws precede
        the model's initial values."""
        if not space_columns:
            return None
        spaces = {}
        for col, info in space_columns.items():
            space = CategorySpace(info['counts'], n_regions=config.labels, dim=config.category_space, seed=random_seed + col,
                                  effects=info['effects'] if config.category_space_init == 'effect' else None,
                                  reconstruction_weight=config.category_space_rec, radius_weight=config.category_space_radius).to(device)
            spaces[real_columns.index(col)] = space.warm_up()
        return spaces

    def build_model(joint_bags):
        spaces = build_spaces()
        inputs = None
        if inputs_factory is not None:
            inputs = inputs_factory(device)
        elif partitions is not None:
            codes = {real_columns.index(col): info['codes'] for col, info in space_columns.items()} if space_columns else None
            inputs = TabularInputs(partitions, config.label_runs, spaces, codes, device)
        return FLAREModel(inputs, n_classes, categorical, cardinalities, config.rules * max(int(joint_bags), 1), config.slots, config.temp,
                          mode, joint_bags, config.allow_repeated_conditions, config.categorical_floor, config.residual, device)

    def train(model, loaders, blocks=0):
        return train_loop(model, loaders[0], loaders[1], config, class_weights, blocks, device, verbose)

    # How the averaged models are trained: see bagging.py. A step of the parallel routes holds about this many elements;
    # the largest intermediates per row and rule are the real variables' conditions and the slots' switched memberships.
    bags, rules_per_bag = config.bags, config.rules
    parallel = config.parallel_bags and bags > 1
    staged, joint_bags, route = False, 1, 'single' if bags == 1 else 'sequential'
    if parallel:
        n_conditions = len(label_run_list(config.labels, config.label_runs))
        width = int(len(categorical) - n_categorical) * n_conditions + len(categorical) * config.slots * 2
        n_train = len(fold.y_train)
        step_elements = batch_size * (n_train + len(fold.y_val)) / n_train * rules_per_bag * bags * width
        # Learned partitions and category spaces cannot be trained by independent blocks: they are learned by the first
        # model and read, frozen, by the others, so only the first one has to train alone.
        shared = []
        if real_columns and (inputs_factory is None or any(p.requires_grad for p in inputs_factory(device).parameters())):
            shared.append('learned fuzzy sets')
        if config.category_space > 0 and fold.category_space:
            shared.append('learned category spaces')
        if shared:
            staged = bags > 2 and step_elements * (bags - 1) / bags <= max_elements
            if verbose:
                print(f'Parallel bags: the {" and ".join(shared)} would be shared by the bags; training '
                      + ('the first bag alone and the others in parallel on what it learned' if staged else 'the bags one after another'), flush=True)
            parallel = False
        elif step_elements > max_elements:
            if verbose:
                print(f'Parallel bags: a step would hold {step_elements / 1e6:.0f}M elements (> {max_elements / 1e6:.0f}M); training the bags one after another', flush=True)
            parallel = False
        else:
            joint_bags, route = bags, 'parallel'

    def single_condition(model, joint):
        if config.single_condition_rules > 0:
            model.restrict_to_single_condition(config.single_condition_rules, bag_size=rules_per_bag if joint > 1 else 0)

    model = build_model(joint_bags)
    single_condition(model, joint_bags)
    if parallel:
        losses, val_accs = train(model, bagging.parallel_bag_loaders(train_loader, val_loader, bags, random_seed, batch_size), blocks=bags)
    else:
        losses, val_accs = train(model, (train_loader, val_loader))

    if staged:
        # The first model (above) learned the crossings and the category spaces. The others read them frozen and train at
        # once as the blocks of one joint model, each on its own split; all the models are then joined in one joint model.
        first, rest = model, bags - 1
        rest_model = build_model(rest)
        rest_model.inputs.inherit(first.inputs)
        single_condition(rest_model, rest)
        train(rest_model, bagging.parallel_bag_loaders(train_loader, val_loader, rest, random_seed, batch_size, first_bag=1), blocks=rest)
        model = build_model(bags)                       # not trained: its blocks are filled from the models above
        model.inputs.inherit(first.inputs)
        single_condition(model, bags)
        bagging.join_blocks(model, [first, rest_model])
        route = 'staged'
    elif route == 'sequential':
        # Every model has its own split of the training rows. The deployed model is the union of their rule bases, which
        # is a BagEnsemble (the points divided by the number of models).
        members = [model]
        for bag in range(1, bags):
            loaders = bagging.sequential_bag_loaders(train_loader, val_loader, bag, random_seed, batch_size)
            member = build_model(1)
            if model.inputs is not None:
                member.inputs.inherit(model.inputs)
            single_condition(member, 1)
            train(member, loaders)
            members.append(member)
        model = BagEnsemble(members)
    time2train = time.time() - tac
    return FitResult(model, fold, train_loader, val_loader, test_loader, config, time2train=time2train, losses=losses,
                     val_accs=val_accs, route=route)


def score_fold(result):
    """The scores of a fitted fold on its test rows (and the training rows): accuracy, macro-F1, balanced accuracy, ROC
    AUC and log-loss of the deployed (discrete) model, the same recalibrated when the configuration asks for it, and the
    size of the rule base. Returns a dict."""
    model, config = result.model, result.config
    test_acc, f1, test_bal_acc = evaluate(model, result.test_loader, 0.0, GradMode.PURE)
    row = {'test_acc': test_acc, 'f1': f1, 'test_bal_acc': test_bal_acc, **probability_metrics(model, result.test_loader)}
    if config.calibrate:                       # recalibrated on the validation rows (the model is unchanged)
        calibration = calibrate_probabilities(model, result.val_loader, config.calibration_min_temperature,
                                              config.calibration_bias_penalty)
        row.update({f'{k}_cal': v for k, v in probability_metrics(model, result.test_loader, calibration).items()},
                   cal_temperature=calibration[0])
    train_acc, train_f1, _ = evaluate(model, result.train_loader, 0.0, GradMode.PURE)
    row.update(train_acc=train_acc, train_f1=train_f1)
    n_rules, n_ants, unique_ants = rule_set_size(model)
    device = next(model.parameters()).device
    deciding, deciding_ants, overlap = rule_set_complexity(model, result.train_loader.dataset.X.to(device))
    row.update(n_rules=n_rules, n_ants=n_ants, unique_ants=unique_ants, n_deciding_rules=deciding,
               n_deciding_ants=deciding_ants, n_deciding_overlap=overlap, time2train=result.time2train)
    return row
