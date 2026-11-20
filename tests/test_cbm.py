"""The concept bottleneck experiments (experiments/cbm, baselines/cbm_heads.py): datasets, detectors, pipeline and heads
on tiny fake datasets, so that nothing here needs the real images."""
import json
import os

import numpy as np
import pytest
import torch

Image = pytest.importorskip('PIL.Image', reason='the concept bottleneck experiments need pillow (pip install -e ".[cbm]")')
pytest.importorskip('torchvision', reason='the concept bottleneck experiments need torchvision (pip install -e ".[cbm]")')

from baselines import cbm_heads
from experiments.cbm import datasets, detector, flare_head, pipeline, run_heads
from flare.inputs import TabularInputs
from flare.model import FLAREModel
from flare.selectors import GradMode


# --- fake datasets -------------------------------------------------------------------------------------------------------

def tiny_jpg(path, seed=0):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    array = np.random.default_rng(seed).integers(0, 255, size=(12, 12, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)


def fake_cub(root, classes=3, per_class=10):
    """A CUB_200_2011 folder: `classes` species of `per_class` images (the first half of each for training), 312 attributes."""
    os.makedirs(os.path.join(root, 'attributes'))
    rng = np.random.default_rng(0)
    images, labels, split, attributes = [], [], [], []
    for c in range(classes):
        for i in range(per_class):
            image_id = len(images) + 1
            images.append(f'{c + 1:03d}.Bird_{c}/img_{i}.jpg')
            tiny_jpg(os.path.join(root, 'images', images[-1]), image_id)
            labels.append(c + 1)
            split.append(int(i < per_class // 2))
            attributes += [f'{image_id} {a} {int(rng.random() < 0.3)} 3 1.0' for a in range(1, 313)]
    write = lambda name, lines: open(os.path.join(root, name), 'w').write('\n'.join(lines) + '\n')       # noqa: E731
    write('images.txt', [f'{i + 1} {p}' for i, p in enumerate(images)])
    write('image_class_labels.txt', [f'{i + 1} {c}' for i, c in enumerate(labels)])
    write('train_test_split.txt', [f'{i + 1} {s}' for i, s in enumerate(split)])
    write('classes.txt', [f'{c + 1} {c + 1:03d}.Bird_{c}' for c in range(classes)])
    write('attributes/attributes.txt', [f'{a} has_attribute_{a}::value' for a in range(1, 313)])
    write('attributes/image_attribute_labels.txt', attributes)


def fake_sun(root, classes=4, per_class=10, attributes=5):
    from scipy.io import savemat
    names = [f'{chr(97 + c)}/scene{c}/sun_{i:04d}.jpg' for c in range(classes) for i in range(per_class)]
    for name in names:
        tiny_jpg(os.path.join(root, 'images', name))
    rng = np.random.default_rng(1)
    votes = rng.integers(0, 4, size=(len(names), attributes)) / 3
    db = os.path.join(root, 'SUNAttributeDB')
    os.makedirs(db)
    def cell(values):                                   # an N x 1 MATLAB cell array of strings, as the database stores them
        array = np.empty((len(values), 1), dtype=object)
        for i, v in enumerate(values):
            array[i, 0] = np.array([v])
        return array
    savemat(os.path.join(db, 'images.mat'), {'images': cell(names)})
    savemat(os.path.join(db, 'attributes.mat'), {'attributes': cell([f'attr/{a}-x' for a in range(attributes)])})
    savemat(os.path.join(db, 'attributeLabels_continuous.mat'), {'labels_cv': votes})
    return names, votes


def fake_derm(root, n=24):
    import pandas as pd
    rng = np.random.default_rng(2)
    diagnoses = ['basal cell carcinoma', 'clark nevus', 'melanoma (less than 0.76 mm)', 'dermatofibroma', 'seborrheic keratosis']
    criteria = {'pigment_network': ['absent', 'typical', 'atypical'], 'streaks': ['absent', 'regular', 'irregular'],
                'pigmentation': ['absent', 'diffuse regular', 'localized irregular'],
                'regression_structures': ['absent', 'blue areas', 'combinations'], 'dots_and_globules': ['absent', 'regular', 'irregular'],
                'blue_whitish_veil': ['absent', 'present'], 'vascular_structures': ['absent', 'arborizing', 'dotted']}
    meta = pd.DataFrame({'case_num': np.arange(1, n + 1), 'diagnosis': [diagnoses[i % 5] for i in range(n)],
                         'derm': [f'D/img{i}.jpg' for i in range(n)], **{k: rng.choice(v, n) for k, v in criteria.items()}})
    os.makedirs(os.path.join(root, 'meta'))
    meta.to_csv(os.path.join(root, 'meta', 'meta.csv'), index=False)
    for split, rows in (('train', range(0, 14)), ('valid', range(14, 19)), ('test', range(19, n))):
        pd.DataFrame({'indexes': list(rows)}).to_csv(os.path.join(root, 'meta', f'{split}_indexes.csv'), index=False)
    for path in meta.derm:
        tiny_jpg(os.path.join(root, 'images', path))
    return meta


# --- manifests ------------------------------------------------------------------------------------------------------------

def test_cub_manifest(tmp_path):
    fake_cub(str(tmp_path))
    m = datasets.build_cub(str(tmp_path))
    assert len(m.concept_names) == len(datasets.KOH_ATTRIBUTES) == 112 and m.concept_names[0] == 'has_attribute_2_value'
    assert m.class_names == ['001.Bird_0', '002.Bird_1', '003.Bird_2']
    sizes = {s: len(m.splits[s]['ids']) for s in datasets.SPLITS}
    assert sizes == {'train': 12, 'val': 3, 'test': 15}                  # 20% of the 5 training images of a class: 1
    ids = [set(m.splits[s]['ids']) for s in datasets.SPLITS]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    again = datasets.build_cub(str(tmp_path))
    assert again.splits == m.splits                                      # the validation draw is seeded
    # the concepts of an image are its own annotation
    first = m.splits['train']['ids'][0]
    lines = [l.split() for l in open(os.path.join(tmp_path, 'attributes', 'image_attribute_labels.txt'))]
    own = {int(a): int(v) for i, a, v, *_ in lines if int(i) == first}
    assert m.splits['train']['C'][0] == [own[a + 1] for a in datasets.KOH_ATTRIBUTES]
    assert os.path.exists(os.path.join(m.image_root, m.splits['train']['paths'][0]))


def test_sun_manifest(tmp_path):
    names, votes = fake_sun(str(tmp_path))
    m = datasets.build_sun(str(tmp_path))
    assert m.concept_names == ['attr_0_x', 'attr_1_x', 'attr_2_x', 'attr_3_x', 'attr_4_x'] and len(m.class_names) == 4
    assert [len(m.splits[s]['ids']) for s in datasets.SPLITS] == [32, 4, 4]          # per class: 8 / 1 / 1
    for split in datasets.SPLITS:
        part = m.splits[split]
        index = np.array(part['ids']) - 1
        assert [names[i] for i in index] == part['paths']
        np.testing.assert_array_equal(part['C'], (votes[index] >= 0.5).astype(int))   # at least two of three votes
        assert all(p.startswith(m.class_names[y]) for p, y in zip(part['paths'], part['y']))
    assert datasets.build_sun(str(tmp_path)).splits == m.splits
    os.remove(os.path.join(tmp_path, 'images', names[3]))
    with pytest.raises(FileNotFoundError):
        datasets.build_sun(str(tmp_path))


def test_derm7pt_manifest(tmp_path):
    meta = fake_derm(str(tmp_path))
    m = datasets.build_derm7pt(str(tmp_path))
    assert len(m.concept_names) == 12 and m.class_names == ['BCC', 'NEV', 'MEL', 'MISC', 'SK']
    assert [len(m.splits[s]['ids']) for s in datasets.SPLITS] == [14, 5, 5]
    train = m.splits['train']
    assert train['y'] == [i % 5 for i in range(14)] and train['paths'] == list(meta.derm[:14])
    column = {n: j for j, n in enumerate(m.concept_names)}
    for row, case in enumerate(meta.iloc[:14].itertuples()):
        assert train['C'][row][column['pigment_network_typical']] == int(case.pigment_network == 'typical')
        assert train['C'][row][column['pigmentation_regular']] == int(case.pigmentation == 'diffuse regular')
        assert train['C'][row][column['vascular_structures_irregular']] == int(case.vascular_structures == 'dotted')
    with pytest.raises(ValueError):
        datasets.derm_group('something else')


def test_manifest_roundtrip_and_limit(tmp_path):
    fake_derm(str(tmp_path / 'd'))
    m = datasets.build_derm7pt(str(tmp_path / 'd'))
    m.save(str(tmp_path / 'm' / 'manifest.json'))
    assert datasets.Manifest.load(str(tmp_path / 'm' / 'manifest.json')).splits == json.loads(json.dumps(m.splits))
    small = m.limited(3)
    assert [len(small.splits[s]['ids']) for s in datasets.SPLITS] == [3, 3, 3]
    files, C = small.rows('train')
    assert len(files) == 3 and C.shape == (3, 12) and C.dtype == np.float32


def toy_export(directory, n=(80, 30, 40), n_concepts=6, n_classes=3, sources=('oracle', 'predicted_seed0'), seed=0):
    """An export in which the class is a function of the concepts; the predicted ones are the oracle ones plus noise."""
    rng = np.random.default_rng(seed)
    names = [f'concept_{j}' for j in range(n_concepts)]
    os.makedirs(directory, exist_ok=True)
    with open(os.path.join(directory, 'concept_names.json'), 'w') as f:
        json.dump(names, f)
    for split, rows in zip(datasets.SPLITS, n):
        C = (rng.random((rows, n_concepts)) < 0.5).astype(np.float32)
        y = (C[:, 0] + C[:, 1] * C[:, 2]).astype(int) % n_classes
        y[:n_classes] = np.arange(n_classes)
        for source in sources:
            soft = C if source == 'oracle' else np.clip(C + 0.3 * rng.normal(size=C.shape), 0, 1)
            datasets.save_split(os.path.join(directory, f'{source}_{split}.npz'), soft, y, np.arange(rows))
    return names


def test_export_roundtrip_and_alignment(tmp_path):
    names = toy_export(str(tmp_path))
    loaded, n_classes, splits = datasets.load_export(str(tmp_path), ['oracle', 'predicted_seed0'])
    assert loaded == names and n_classes == 3 and splits['train']['oracle'].shape == (80, 6)
    assert datasets.has_sources(str(tmp_path), ['oracle']) and not datasets.has_sources(str(tmp_path), ['crossfit_seed0'])
    with pytest.raises(FileNotFoundError):
        datasets.load_export(str(tmp_path), ['crossfit_seed0'])
    other = np.load(os.path.join(tmp_path, 'oracle_test.npz'))
    np.savez(os.path.join(tmp_path, 'predicted_seed0_test.npz'), C=other['C'], y=other['y'], ids=other['ids'][::-1])
    with pytest.raises(ValueError):
        datasets.load_export(str(tmp_path), ['oracle', 'predicted_seed0'])


# --- FLARE as a head --------------------------------------------------------------------------------------------------------

def test_concept_inputs_contract():
    inputs = flare_head.ConceptInputs(4)
    c = torch.tensor([[0.0, 0.25, 0.75, 1.0]])
    m = inputs(c)
    assert m.shape == (1, 4, 2) and inputs.n_variables == 4 and inputs.n_conditions == 2
    torch.testing.assert_close(m[0, :, 1], c[0])                         # TRUE has truth c
    torch.testing.assert_close(m[0, :, 0], 1 - c[0])                     # FALSE has truth 1 - c
    assert inputs.crossing_parameters() == [] and inputs.space_parameters() == [] and not list(inputs.parameters())
    assert inputs.condition_text(2, 1, 'streaks') == 'streaks IS TRUE' and inputs.condition_text(2, 0, 'streaks') == 'streaks IS FALSE'
    model = FLAREModel(inputs, 3, np.zeros(4, dtype=bool), [], 6, 2, 5.0, GradMode.GRAFT)
    scores = model(c)[0]
    assert scores.shape == (1, 3)


SMALL = dict(rules=12, single_condition_rules=8, bags=2, epochs=3, labels=5, cuda_graph=False)


def small_flare(**changes):
    return flare_head.concept_config(**{**SMALL, **changes})


def test_flare_head_trains_and_is_reproducible(tmp_path):
    names = toy_export(str(tmp_path))
    _, _, splits = datasets.load_export(str(tmp_path), ['oracle', 'predicted_seed0'])
    results = [flare_head.fit_flare(names, splits, 'oracle', 'predicted_seed0', 33, small_flare(), 'cpu') for _ in range(2)]
    a, b = results
    assert a.route == 'parallel'                                         # two small models train at once
    y = splits['test']['y']
    scores = [flare_head.score_flare(r, splits['test']['predicted_seed0'], y) for r in results]
    assert scores[0] == scores[1] and 0 < scores[0][1] <= 1
    assert flare_head.printed_rules(a, names) == flare_head.printed_rules(b, names)
    text = flare_head.printed_rules(a, names)
    assert len(text.splitlines()) == 24 and all(' IS TRUE' in line or ' IS FALSE' in line for line in text.splitlines())
    counts = flare_head.rule_counts(a)
    assert 0 < counts['n_rules'] <= 24 and counts['n_deciding_rules'] <= counts['n_rules']
    # one model per bag, one after another, gives another route with the same interface
    seq = flare_head.fit_flare(names, splits, 'oracle', 'predicted_seed0', 33, small_flare(parallel_bags=False), 'cpu')
    assert seq.route == 'sequential'


def test_evaluation_loaders_keep_a_fixed_order(tmp_path):
    from flare.data import fold_loaders
    names = toy_export(str(tmp_path))
    _, _, splits = datasets.load_export(str(tmp_path), ['oracle', 'predicted_seed0'])
    train, val, test = fold_loaders(flare_head.concept_fold(splits, 'oracle', 'predicted_seed0'), 16, 'cpu',
                                    fixed_order_evaluation=True)
    first = next(iter(val))[1]
    assert torch.equal(first, torch.as_tensor(splits['val']['y'][:len(first)])) and torch.equal(next(iter(test))[1], torch.as_tensor(splits['test']['y'][:512]))
    again = next(iter(val))[1]
    assert torch.equal(first, again)                                     # the same rows in the same order on every pass


def test_flare_head_batch_size_follows_the_training_rows(tmp_path):
    names = toy_export(str(tmp_path), n=(400, 30, 40))
    _, _, splits = datasets.load_export(str(tmp_path), ['oracle', 'predicted_seed0'])
    result = flare_head.fit_flare(names, splits, 'oracle', 'predicted_seed0', 1, small_flare(bags=1, epochs=1), 'cpu')
    assert result.train_loader.batch_size == 32                          # ceil(400 / 50) = 8, below the minimum
    toy_export(str(tmp_path / 'b'), n=(2000, 30, 40))
    _, _, splits = datasets.load_export(str(tmp_path / 'b'), ['oracle', 'predicted_seed0'])
    result = flare_head.fit_flare(names, splits, 'oracle', 'predicted_seed0', 1, small_flare(bags=1, epochs=1), 'cpu')
    assert result.train_loader.batch_size == 40                          # ceil(2000 / 50)


# --- the other heads ----------------------------------------------------------------------------------------------------------

def test_head_settings():
    assert cbm_heads.rule_count(5) == 250 and cbm_heads.rule_count(200) == 250 and cbm_heads.rule_count(717) == 767
    assert cbm_heads.runs_on('DR-Net', 5) and not cbm_heads.runs_on('DR-Net', 200) and cbm_heads.runs_on('RRL', 717)
    with pytest.raises(KeyError):
        cbm_heads.ConceptHead('nope', 3, 0)
    # the two macro-F1: over the classes of the labels, or of the labels and the predictions
    y, p = np.array([0, 0, 1, 1]), np.array([0, 2, 1, 1])
    assert cbm_heads.ConceptHead('LR', 3, 0).macro_f1(y, p) == pytest.approx((2 / 3 + 1) / 2)
    assert cbm_heads.ConceptHead('LogicCBM', 3, 0).macro_f1(y, p) == pytest.approx((2 / 3 + 1 + 0) / 3)


@pytest.mark.parametrize('head', ['LR', 'GradTree', 'LogicCBM', 'DCR'])
def test_heads_learn_a_toy_problem(head, tmp_path):
    toy_export(str(tmp_path), n=(300, 60, 80))
    names, n_classes, splits = datasets.load_export(str(tmp_path), ['oracle'])
    model = cbm_heads.ConceptHead(head, n_classes, 3, 'cpu')
    model.fit(splits['train']['oracle'], splits['train']['y'], splits['val']['oracle'], splits['val']['y'])
    predicted = model.predict(splits['test']['oracle'])
    assert predicted.shape == splits['test']['y'].shape
    assert model.macro_f1(splits['test']['y'], predicted) > 0.5


def test_crisp_heads_read_the_concepts_cut_at_half():
    head = cbm_heads.ConceptHead('RRL', 3, 0)
    np.testing.assert_array_equal(head._inputs(np.array([[0.2, 0.5, 0.51]])), [[0.0, 0.0, 1.0]])
    soft = cbm_heads.ConceptHead('LogicCBM', 3, 0)
    np.testing.assert_array_equal(soft._inputs(np.array([[0.2, 0.5, 0.51]])), [[0.2, 0.5, 0.51]])


# --- the runner -----------------------------------------------------------------------------------------------------------------

def test_run_heads_writes_resumable_rows(tmp_path, capsys, monkeypatch):
    sources = ['oracle', 'predicted_seed0', 'predicted_seed1', 'crossfit_seed0', 'crossfit_seed1']
    toy_export(str(tmp_path / 'export'), sources=sources)
    full = flare_head.concept_config
    monkeypatch.setattr(flare_head, 'concept_config', lambda **kw: full(**{**SMALL, **kw}))
    args = ['--dataset', 'derm7pt', '--export', str(tmp_path / 'export'), '--heads', 'FLARE', 'LR', '--seeds', '33', '34',
            '--device', 'cpu', '--out', str(tmp_path / 'out')]
    run_heads.main(args)
    out = capsys.readouterr().out
    assert '[4/4]' in out
    import csv
    rows = list(csv.DictReader(open(tmp_path / 'out' / 'derm7pt_independent.csv')))
    assert [(r['head'], r['seed'], r['detector_seed']) for r in rows] == [('FLARE', '33', '0'), ('LR', '33', '0'), ('FLARE', '34', '1'), ('LR', '34', '1')]
    assert all(r['train_source'] == 'oracle' and r['test_source'] == f'predicted_seed{r["detector_seed"]}' for r in rows)
    assert all(0 < float(r['f1']) <= 1 for r in rows) and rows[0]['n_rules'] and not rows[1]['n_rules']
    assert os.path.exists(tmp_path / 'out' / 'rules' / 'derm7pt_independent_FLARE_seed33.txt')
    run_heads.main(args)                                                  # nothing is run twice
    assert 'already in' in capsys.readouterr().out
    assert len(list(csv.DictReader(open(tmp_path / 'out' / 'derm7pt_independent.csv')))) == 4
    run_heads.main(args + ['--regime', 'sequential'])
    rows = list(csv.DictReader(open(tmp_path / 'out' / 'derm7pt_sequential.csv')))
    assert [r['train_source'] for r in rows][::2] == ['crossfit_seed0', 'crossfit_seed1']


# --- detectors and the pipeline -------------------------------------------------------------------------------------------------

def test_image_tensor_follows_the_original_normalisation():
    t = detector.image_to_tensor(Image.new('RGB', (10, 10), (255, 0, 51)))
    assert t.shape == (3, 299, 299)
    torch.testing.assert_close(t[:, 0, 0], torch.tensor([(1.0 - 0.5) / 2, (0.0 - 0.5) / 2, (0.2 - 0.5) / 2]))


def test_concept_network_has_one_output_per_concept():
    net = detector.ConceptInception(7, pretrained=False)
    assert net.concepts.weight.shape == (7, 2048) and net.concepts.weight.abs().max() <= 0.2
    assert net.backbone.transform_input and not net.backbone.aux_logits
    net.eval()
    with torch.no_grad():
        assert net(torch.zeros(2, 3, 299, 299)).shape == (2, 7)


def test_koh_detector_trains_saves_and_predicts(tmp_path):
    paths = []
    for i in range(6):
        paths.append(str(tmp_path / f'im{i}.jpg'))
        tiny_jpg(paths[-1], i)
    C = (np.random.default_rng(0).random((6, 3)) < 0.5).astype(np.float32)
    rows, recipe = detector.Rows(paths, C), detector.RECIPES['cub']
    messages = []
    model = detector.fit_detector(recipe, rows, None, 0, 'cpu', epochs=1, batch_size=3, pretrained=False, log=messages.append)
    assert len(messages) == 1 and messages[0].startswith('seed=0 epoch=1 loss=')
    P = detector.predict(recipe, model, rows, 'cpu', batch_size=3)
    assert P.shape == (6, 3) and P.min() >= 0 and P.max() <= 1
    detector.save_detector(str(tmp_path / 'd.pt'), model, 'cub', ['a', 'b', 'c'], 0)
    loaded, recipe_back, names = detector.load_detector(str(tmp_path / 'd.pt'))
    assert recipe_back == recipe and names == ['a', 'b', 'c']
    np.testing.assert_allclose(detector.predict(recipe, loaded, rows, 'cpu', batch_size=3), P, atol=1e-6)
    assert [r.epochs for r in detector.RECIPES.values()] == [100, 30, 40]


@pytest.mark.slow
def test_pipeline_from_images_to_heads(tmp_path):
    """prepare -> train -> export -> crossfit on a fake SUN folder, then a head on the result."""
    fake_sun(str(tmp_path / 'sun'))
    work = str(tmp_path / 'work')
    pipeline.main(['prepare', 'sun', '--root', str(tmp_path / 'sun'), '--work', work])
    common = ['--work', work, '--seeds', '0', '--device', 'cpu', '--workers', '0']
    pipeline.main(['train', *common, '--epochs', '1', '--batch_size', '8', '--no_pretrained'])
    pipeline.main(['train', *common, '--epochs', '1', '--batch_size', '8', '--no_pretrained'])          # resumes: nothing to do
    pipeline.main(['export', *common])
    pipeline.main(['crossfit', *common, '--epochs', '1', '--batch_size', '8', '--no_pretrained', '--folds', '2'])
    names, n_classes, splits = datasets.load_export(os.path.join(work, 'export'), ['oracle', 'predicted_seed0', 'crossfit_seed0'])
    assert len(names) == 5 and n_classes == 4
    assert splits['train']['predicted_seed0'].shape == (32, 5) and not np.isnan(splits['train']['crossfit_seed0']).any()
    np.testing.assert_array_equal(splits['test']['crossfit_seed0'], splits['test']['predicted_seed0'])
    assert not np.array_equal(splits['train']['crossfit_seed0'], splits['train']['predicted_seed0'])   # other detectors
    row, _ = run_heads.run_head('LR', names, n_classes, splits, 'crossfit_seed0', 'predicted_seed0', 33, 'cpu')
    assert 0 <= row['f1'] <= 1
