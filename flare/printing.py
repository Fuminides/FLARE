"""Printing the rules.

A rule reads  IF <condition> AND <condition> ... THEN Class c  (w=...):  the conditions are what the rule tests (the
labels of real variables with the point where they cut, the categories of categorical ones, the groups of categories
of a category space); the class is the one the rule adds most to, and w is that point as a share of the sum of the
absolute points of the rule. The model itself adds up the points of all its rules on every class, so the class shown
is the rule's main effect, not an exclusive vote.
"""

from .model import BagEnsemble


def rules_text(model, feat_names=None, class_names=None, category_names=None):
    """The rule base of a model as text, one rule per line, in the order of the rules. Averaged models trained one after
    another (BagEnsemble) print every model after a header line "# model k of B"; averaged models trained together are
    one model whose rules follow each other (rules 1-R of the first model, then the next R, ...).

    feat_names: the name of each column of the data (default x0, x1, ...); class_names: the name of each class
    (default "Class c"); category_names: {column: [name of each category code]} for categorical columns (including
    those with a category space; without names, a categorical condition shows the code)."""
    if isinstance(model, BagEnsemble):
        B = len(model.members)
        return '\n'.join(f'# model {k} of {B}\n' + rules_text(member, feat_names, class_names, category_names)
                         for k, member in enumerate(model.members, 1))
    rules, rule_weights, consequents = model.extract_rules()
    category_names = category_names or {}
    real_position = {column: i for i, column in enumerate(model.real_columns)}
    lines = []
    for r in range(rules.shape[0]):
        conditions = []
        for column in range(rules.shape[1]):
            condition = rules[r, column]
            if condition == -1:
                continue
            name = feat_names[column] if feat_names is not None else f'x{column}'
            if column in real_position:
                conditions.append(model.inputs.condition_text(real_position[column], int(condition), name, category_names.get(column)))
            else:
                labels = category_names.get(column)
                conditions.append(f'{name} IS {labels[int(condition)] if labels is not None else condition}')
        c = int(consequents[r])
        class_name = class_names[c] if class_names is not None else f'Class {c}'
        lines.append(f"IF {' AND '.join(conditions) if conditions else '<no conditions>'} THEN {class_name}  (w={rule_weights[r, c]:.3f})")
    return '\n'.join(lines)


def print_rules(model, save_path=None, **kwargs):
    text = rules_text(model, **kwargs)
    print(text)
    if save_path:
        with open(save_path, 'w') as f:
            f.write(text + '\n')
    return text
