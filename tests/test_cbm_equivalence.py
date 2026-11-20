"""Reruns of reference runs of the concept bottleneck heads (slow).

Like tests/test_equivalence.py, the reference directory is given by the environment variable FLARE_REFERENCE_DIR. Each
case <name> is a folder <FLARE_REFERENCE_DIR>/cbm/<name>/ with

    export/          the concept export the run read (experiments/cbm/datasets.py describes the layout)
    reference.json   seed, number of epochs, training and test sources, the accuracy and macro-F1 of FLARE on the test
                     split of each source, its rule counts, and the same scores of the other heads on the test source
    rules.txt        the printed rules of FLARE

The tests train FLARE and the heads on the CPU (two threads, tests/conftest.py) and compare: FLARE exactly (accuracy,
macro-F1, number of rules, distinct conditions, deciding rules and every printed rule), the other heads exactly, and
NeuRules within a tolerance (the reference used another implementation). A case is skipped if the reference is missing;
a head whose authors' code is not cloned is skipped.
"""
import glob
import json
import os

import pytest

from baselines import cbm_heads
from experiments.cbm import datasets, flare_head

REFERENCE = os.environ.get('FLARE_REFERENCE_DIR')

pytestmark = [pytest.mark.slow, pytest.mark.skipif(not REFERENCE, reason='FLARE_REFERENCE_DIR is not set')]

CASES = sorted(os.path.basename(p) for p in glob.glob(os.path.join(REFERENCE or '', 'cbm', '*')))
TOLERANT = {'NeuRules': 0.03}                    # absolute tolerance on macro-F1 and accuracy


def load(case):
    folder = os.path.join(REFERENCE, 'cbm', case)
    reference = json.load(open(os.path.join(folder, 'reference.json')))
    sources = sorted({reference['train_source'], reference['test_source'], *reference['flare']['scores']})
    names, n_classes, splits = datasets.load_export(os.path.join(folder, 'export'), sources)
    return folder, reference, names, n_classes, splits


@pytest.mark.parametrize('case', CASES)
def test_flare_head_matches_reference(case):
    folder, reference, names, _, splits = load(case)
    config = flare_head.concept_config(epochs=reference['epochs'])
    result = flare_head.fit_flare(names, splits, reference['train_source'], reference['test_source'], reference['seed'], config, 'cpu')
    assert result.route == reference['route']
    expected = reference['flare']
    for source, scores in expected['scores'].items():
        acc, f1 = flare_head.score_flare(result, splits['test'][source], splits['test']['y'])
        assert (acc, f1) == (scores['acc'], scores['macro_f1']), source
    counts = flare_head.rule_counts(result)
    assert {k: int(v) for k, v in counts.items()} == expected['rule_counts']
    text = flare_head.printed_rules(result, names)
    if result.route == 'sequential':     # the reference lists only the first of the models trained one after another
        text = text.split('\n# model 2 of ')[0].split('\n', 1)[1]
    assert text == open(os.path.join(folder, 'rules.txt')).read().rstrip('\n')


@pytest.mark.parametrize('case', CASES)
def test_other_heads_match_reference(case):
    _, reference, _, n_classes, splits = load(case)
    train, test = reference['train_source'], reference['test_source']
    for head, expected in reference['heads'].items():
        model = cbm_heads.ConceptHead(head, n_classes, reference['seed'], 'cpu')
        try:
            model.fit(splits['train'][train], splits['train']['y'], splits['val'][train], splits['val']['y'])
        except ImportError as error:                  # the authors' code is not cloned
            print(f'{head} skipped: {error}')
            continue
        predicted = model.predict(splits['test'][test])
        f1, acc = model.macro_f1(splits['test']['y'], predicted), float((predicted == splits['test']['y']).mean())
        tolerance = TOLERANT.get(head, 1e-12)
        assert f1 == pytest.approx(expected['macro_f1'], abs=tolerance, rel=0), head
        assert acc == pytest.approx(expected['acc'], abs=tolerance, rel=0), head
