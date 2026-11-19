"""Frozen spatial-map corrections for paper Experiment V; CPU replay of saved exports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sqlite3
from contextlib import closing

import numpy as np
from PIL import Image
from skimage.transform import resize
import torch

from spatial_cbm.training.data import digest, save_json


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')


def classifier(cfg, head_dir, identity, classes, concepts, nec):
    root, head_dir = Path(cfg.training.output), Path(head_dir)
    meta_path = head_dir/'classifier_identity.json'
    meta = read(meta_path)
    norm_path = root/'final_layer_normalization.pt'
    if (meta['checkpoint_sha256'] != identity['checkpoint_sha256'] or
            meta['feature_manifest_sha256'] != digest(root/'features_complete.json') or
            meta['normalization_sha256'] != digest(norm_path)):
        raise ValueError('Classifier checkpoint, features or normalization changed')
    files = [meta_path, root/'best.pt', root/'features_complete.json', norm_path]
    if cfg.model.checkpoint:
        files.append(Path(cfg.model.checkpoint))
    norm = torch.load(norm_path, map_location='cpu', weights_only=True)
    mean, std = norm['mean'].float(), norm['std'].float()
    values = []
    for prefix in ('W_g', 'b_g'):
        path = head_dir/f'{prefix}@NEC={nec}.pt'
        if meta['head_files'].get(path.name) != digest(path):
            raise ValueError('Missing or changed NEC classifier')
        values.append(torch.load(path, map_location='cpu', weights_only=True).float())
        files.append(path)
    weight, bias = values
    if (mean.shape != (concepts,) or std.shape != mean.shape or not (std > 0).all()
            or weight.shape != (classes, concepts) or bias.shape != (classes,)
            or not all(torch.isfinite(v).all() for v in (mean, std, weight, bias))):
        raise ValueError('Invalid classifier dimensions or normalization')
    if int((weight.abs() > 1e-5).sum()) != nec*classes:
        raise ValueError('Exact NEC mismatch')
    return dict(weight=weight.numpy(), bias=bias.numpy(), mean=mean.numpy(), std=std.numpy()), {
        str(p): digest(p) for p in files}


def missed(scores, calibration):
    scores = np.asarray(scores)
    macro = calibration['metrics'].get('concept_macro')
    if macro is not None:
        frozen = macro['calibration']
        policies = np.asarray(frozen['policies'])
        thresholds = np.asarray([0. if t is None else t for t in frozen['thresholds']], dtype=float)
        missing = np.asarray([t is None for t in frozen['thresholds']])
        if (frozen.get('score') != 'raw_logit' or policies.shape != scores.shape or thresholds.shape != scores.shape
                or not np.isin(policies, ['absent', 'present', 'threshold']).all()
                or not np.isfinite(thresholds).all() or np.any(missing == (policies == 'threshold'))):
            raise ValueError('Invalid frozen per-concept calibration')
        return (policies == 'absent') | ((policies == 'threshold') & (scores < thresholds))
    concept = calibration['metrics']['concept']
    if concept['score'] != 'sigmoid(concept_score)' or concept['comparison'] != '>=':
        raise ValueError('Unknown legacy threshold score space')
    threshold = concept['threshold']
    if not np.isfinite(threshold) or not 0 <= threshold <= 1.000001:
        raise ValueError('Invalid legacy concept threshold')
    return torch.from_numpy(np.asarray(scores)).sigmoid().numpy() < threshold


def pool(maps, mode):
    flat = np.asarray(maps, dtype=np.float64).reshape(len(maps), -1)
    maximum = flat.max(1, keepdims=True)
    exponential = np.exp(flat-maximum)
    if mode == 'lse':
        return maximum[:, 0] + np.log(exponential.mean(1))
    if mode == 'salf_softmax':
        return (flat*exponential).sum(1)/exponential.sum(1)
    raise ValueError('Unsupported native pooling')


def native_masks(mask, size):
    if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or not 1 <= size <= mask.shape[0]:
        raise ValueError('Mask must be a square input crop with a valid native grid')
    # Integrate pixel area exactly, including 12x12 grids that do not divide 224.
    pixels = np.arange(mask.shape[0], dtype=float)
    edges = np.linspace(0, mask.shape[0], size+1)
    overlap = np.maximum(0., np.minimum(edges[1:, None], pixels+1)-np.maximum(edges[:-1, None], pixels))
    overlap /= mask.shape[0]/size
    weighted = np.einsum('gi,ij->gj', overlap, mask.astype(float), optimize=False)
    occupancy = np.einsum('gj,hj->gh', weighted, overlap, optimize=False).astype('float32')
    # This is SALF's demo resize/support convention, not occupancy thresholding.
    support = resize(mask.astype(float), (size, size), order=1, mode='reflect', anti_aliasing=True) > 0
    return occupancy, support.astype('float32')


def export(cfg, head_dir, feedback, output, split='test', nec=5, budget=5):
    from spatial_cbm.training.data import Samples
    from spatial_cbm.training.model import load_selected
    if split not in {'test', 'validation'} or not 1 <= budget <= 5 or nec < 1:
        raise ValueError('Use validation/test, a positive NEC and an edit budget from one to five')
    if cfg.model.method not in {'sg_cbm', 'salf_cbm'}:
        raise ValueError('Intervention requires native SpG-CBM or SALF-CBM maps')
    if cfg.model.method == 'sg_cbm' and cfg.model.spatial_pooling != 'lse':
        raise ValueError('Paper intervention exporter supports SpG-CBM LSE pooling only')
    output, feedback = Path(output), Path(feedback).resolve()
    if output.exists():
        raise FileExistsError(output)
    backbone, head, bank, data, identity = load_selected(cfg)
    columns = {name: j for j, name in enumerate(bank)}
    tensors, files = classifier(cfg, head_dir, identity, data['classes'], len(bank), nec)
    calibration_path = Path(cfg.training.output)/'calibration.json'
    calibration = read(calibration_path)
    if calibration['identity'] != identity or calibration['cohort'] != 'validation':
        raise ValueError('Require matching validation-only concept calibration')
    files.update({str(calibration_path): digest(calibration_path), str(feedback): digest(feedback)})
    specification = read(feedback)
    source, rows = specification['source'], specification['images']
    if source not in {'gdino', 'sam3', 'human'} or not rows:
        raise ValueError('Feedback needs a supported source and a nonempty full cohort')
    if source == 'human' and cfg.dataset != 'partimagenetpp':
        raise ValueError('Human-part protocol requires PartImageNet++')
    class_concepts = specification.get('class_concepts', {})
    if cfg.dataset != 'partimagenetpp' and not class_concepts:
        raise ValueError('Provide the training-derived true-class concept vocabulary')
    if any(not isinstance(v, list) or any(c not in bank for c in v) for v in class_concepts.values()):
        raise ValueError('Class vocabulary contains unknown concepts')
    dataset = Samples(cfg, split, data, concepts=False)
    with closing(sqlite3.connect(f'file:{cfg.data.prepared}/{dataset.database}?mode=ro', uri=True)) as db:
        lookup = {position: (name, label) for position, name, label in
                  db.execute('SELECT position,id,class_id FROM samples')}
    records = {lookup[int(position)][0]: (i, lookup[int(position)][1])
               for i, position in enumerate(dataset.indices)}
    names = [r['id'] for r in rows]
    if len(set(names)) != len(names) or not set(names) <= records.keys():
        raise ValueError('Feedback roster contains duplicate IDs or images outside the requested split')
    for row in rows:
        concepts = [e['concept'] for e in row['edits']]
        if len(set(concepts)) != len(concepts) or not set(concepts) <= set(bank):
            raise ValueError('Feedback contains duplicate or unknown image-concept pairs')
        if cfg.dataset != 'partimagenetpp' and str(records[row['id']][1]) not in class_concepts:
            raise ValueError('Class vocabulary does not cover the cohort')
        for edit in row['edits']:
            path = (feedback.parent/edit['mask']).resolve()
            files[str(path)] = digest(path)
            confidence = edit.get('confidence', 1. if source == 'human' else None)
            if confidence is None or not np.isfinite(confidence) or not 0 <= confidence <= 1:
                raise ValueError('Feedback needs finite confidence in [0,1]')
    spatial_weight = (1. if cfg.model.method == 'salf_cbm' else
                      1-float(head.mixing_alpha()) if cfg.model.fusion == 'convex' else float(cfg.model.alpha))
    mode = 'salf_softmax' if cfg.model.method == 'salf_cbm' else 'lse'
    output.mkdir(parents=True, exist_ok=False)
    scores, labels, maps, occupancy, supports, pairs, input_hashes = [], [], [], [], [], [], []
    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    backbone.float().eval(); head.float().eval()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    grid = None
    try:
        with torch.inference_mode(), torch.autocast(torch.device(cfg.training.device).type, enabled=False):
            for i, row in enumerate(rows):
                position, label = records[row['id']]
                image = dataset[position][0][None].to(cfg.training.device).float()
                input_hashes.append(hashlib.sha256(image.cpu().numpy().tobytes()).hexdigest())
                if cfg.training.channels_last:
                    image = image.contiguous(memory_format=torch.channels_last)
                prediction = head(backbone(image))
                raw = prediction['final_logits'][0].cpu().numpy()
                native = prediction['spatial_maps'][0].float().cpu().numpy()
                if not np.isfinite(raw).all() or not np.isfinite(native).all():
                    raise ValueError('Nonfinite model outputs')
                grid = native.shape[-1]
                pooled = pool(native, mode)
                expected = prediction.get('spatial_logits', prediction['final_logits'])[0].cpu().numpy()
                np.testing.assert_allclose(pooled, expected, rtol=2e-5, atol=2e-5)
                scores.append(raw); labels.append(label)
                absent = missed(raw, calibration)
                candidates = sorted(enumerate(row['edits']), key=lambda e: (-e[1].get('confidence', 1.), e[0]))
                selected = 0
                for rank, edit in candidates:
                    column = columns[edit['concept']]
                    confidence = edit.get('confidence', 1.)
                    if (not absent[column] or (source == 'gdino' and confidence <= .15) or
                            (source == 'sam3' and confidence < .5)):
                        continue
                    if cfg.dataset != 'partimagenetpp' and edit['concept'] not in class_concepts[str(label)]:
                        continue
                    path = (feedback.parent/edit['mask']).resolve()
                    with Image.open(path) as im:
                        mask = np.asarray(im.convert('L'))
                    if mask.shape != (cfg.data.input_size, cfg.data.input_size) or not np.isin(mask, [0, 1, 255]).all():
                        raise ValueError('Feedback masks must be binary and aligned to the model input crop')
                    if not mask.any():
                        continue
                    q, support = native_masks(mask > 0, grid)
                    maps.append(native[column]); occupancy.append(q); supports.append(support)
                    pairs.append(dict(image=i, concept=column, feedback_order=rank, confidence=confidence))
                    selected += 1
                    if selected == budget:
                        break
                print(json.dumps(dict(completed=i+1, total=len(rows), edits=len(pairs))), flush=True)
        if any(digest(path) != expected for path, expected in files.items()):
            raise ValueError('Input files changed during export')
        arrays = dict(**tensors, scores=np.stack(scores), labels=np.asarray(labels, dtype='int64'),
                      maps=np.asarray(maps, dtype='float32').reshape(-1, grid, grid),
                      occupancy=np.asarray(occupancy, dtype='float32').reshape(-1, grid, grid),
                      support=np.asarray(supports, dtype='float32').reshape(-1, grid, grid),
                      image_indices=np.array([p['image'] for p in pairs], dtype='int64'),
                      concept_indices=np.array([p['concept'] for p in pairs], dtype='int64'))
        np.savez_compressed(output/'arrays.npz', **arrays)
        common = dict(model=identity, dataset=cfg.dataset, method=cfg.model.method, concepts=bank,
                      nec=nec, budget=budget, pooling=mode, spatial_weight=spatial_weight,
                      calibration_sha256=digest(calibration_path), source=source,
                      class_vocabulary=class_concepts, classifier_files={Path(p).name: h for p, h in files.items()
                          if Path(p).parent == Path(head_dir)}, grid=grid,
                      input_size=cfg.data.input_size, resize_size=cfg.data.resize_size,
                      evaluator_sha256=digest(__file__))
        protocol = dict(identity=common, split=split, image_ids=names, input_sha256=input_hashes,
                        files=files, pairs=pairs, arrays_sha256=digest(output/'arrays.npz'),
                        concept_selection='Missed present true-class concepts; confidence descending then feedback order',
                        limitations='Simulated corrections; mask provenance and crop alignment are supplied by the caller')
        save_json(output/'export.json', protocol)
        save_json(output/'complete.json', dict(export_sha256=digest(output/'export.json')))
        return protocol
    finally:
        dataset.close()
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def load_export(root):
    root = Path(root)
    if read(root/'complete.json')['export_sha256'] != digest(root/'export.json'):
        raise ValueError('Export metadata changed')
    metadata = read(root/'export.json')
    if metadata['arrays_sha256'] != digest(root/'arrays.npz'):
        raise ValueError('Export arrays changed')
    with np.load(root/'arrays.npz', allow_pickle=False) as archive:
        arrays = {k: archive[k] for k in archive.files}
    validate_arrays(metadata, arrays)
    return metadata, arrays


def validate_arrays(metadata, arrays):
    raw, labels = arrays['scores'], arrays['labels']
    if raw.ndim != 2 or not all(raw.shape) or labels.shape != (len(raw),):
        raise ValueError('Invalid exported score or label dimensions')
    n, c = raw.shape
    names = metadata['image_ids']
    if len(names) != n or len(set(names)) != n:
        raise ValueError('Export roster dimensions or uniqueness differ')
    w, b, std = arrays['weight'], arrays['bias'], arrays['std']
    if (w.ndim != 2 or w.shape[1] != c or b.shape != (len(w),) or
            std.shape != (c,) or arrays['mean'].shape != (c,) or not (std > 0).all()):
        raise ValueError('Invalid exported classifier dimensions or normalization')
    if labels.dtype.kind not in 'iu' or np.any((labels < 0) | (labels >= len(w))):
        raise ValueError('Invalid exported class labels')
    ri, ci = arrays['image_indices'], arrays['concept_indices']
    if (ri.ndim != 1 or ci.shape != ri.shape or ri.dtype.kind not in 'iu' or ci.dtype.kind not in 'iu'
            or np.any((ri < 0) | (ri >= n)) or np.any((ci < 0) | (ci >= c))
            or len(set(zip(ri.tolist(), ci.tolist()))) != len(ri)):
        raise ValueError('Invalid or duplicate exported intervention pairs')
    if np.any(np.bincount(ri, minlength=n) > metadata['identity']['budget']):
        raise ValueError('Export exceeds the edit budget')
    size = metadata['identity']['grid']
    for key in ('maps', 'occupancy', 'support'):
        if arrays[key].shape != (len(ri), size, size):
            raise ValueError('Invalid native map or mask dimensions')
    if (np.any((arrays['occupancy'] < 0) | (arrays['occupancy'] > 1)) or
            not np.isin(arrays['support'], [0, 1]).all()):
        raise ValueError('Invalid mask occupancy or support')
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite intervention export')


def compute(metadata, arrays, rule, beta=1.):
    if rule not in {'replace', 'additive'} or not math.isfinite(beta) or beta < 0:
        raise ValueError('Unknown edit rule or invalid beta')
    raw = np.asarray(arrays['scores'], dtype=np.float64)
    changed = raw.copy()
    ri, ci = arrays['image_indices'], arrays['concept_indices']
    if not all(np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Nonfinite intervention export')
    if len(ri):
        q = np.clip(arrays['occupancy'].astype(float), .001, .999)
        edited = np.log(q/(1-q)) if rule == 'replace' else arrays['maps'].astype(float)+beta*arrays['support']
        delta = metadata['identity']['spatial_weight']*(pool(edited, metadata['identity']['pooling']) -
                                                       pool(arrays['maps'], metadata['identity']['pooling']))
        changed[ri, ci] += delta
    logits = lambda x: ((x-arrays['mean'])/arrays['std']) @ arrays['weight'].astype(float).T+arrays['bias']
    before, after = logits(raw).argmax(1), logits(changed).argmax(1)
    correct, corrected = before == arrays['labels'], after == arrays['labels']
    return dict(images=len(raw), edits=len(ri), edited_images=len(np.unique(ri)), rule=rule,
                beta=beta if rule == 'additive' else None,
                base_accuracy=float(correct.mean()*100), edited_accuracy=float(corrected.mean()*100),
                gain_pp=float((corrected.astype(float)-correct).mean()*100),
                fixed=int((~correct & corrected).sum()), harmed=int((correct & ~corrected).sum()),
                predictions_before=before.tolist(), predictions_after=after.tolist())


def calibrate(root, output, betas):
    if Path(output).exists():
        raise FileExistsError(output)
    meta, arrays = load_export(root)
    if meta['split'] != 'validation' or not betas:
        raise ValueError('Beta selection requires a validation export and a prespecified grid')
    results = [compute(meta, arrays, 'additive', b) for b in sorted(set(betas))]
    best = max(results, key=lambda r: (r['edited_accuracy'], -r['beta']))
    value = dict(identity=meta['identity'], image_ids=meta['image_ids'], beta=best['beta'],
                 export_sha256=digest(Path(root)/'export.json'), split='validation',
                 search=[{k: v for k, v in r.items() if not k.startswith('predictions')} for r in results],
                 objective='maximum validation accuracy; smallest beta on ties')
    write_new(output, value)
    return value


def evaluate(root, output, rule='replace', calibration=None):
    if Path(output).exists():
        raise FileExistsError(output)
    meta, arrays = load_export(root)
    if meta['split'] != 'test':
        raise ValueError('Final evaluation requires a test export')
    beta = 1.
    if rule == 'additive':
        if calibration is None:
            raise ValueError('Additive evaluation requires frozen validation beta calibration')
        frozen = read(calibration)
        if frozen['identity'] != meta['identity'] or frozen['split'] != 'validation':
            raise ValueError('Beta calibration belongs to another model or protocol')
        if set(frozen['image_ids']) & set(meta['image_ids']):
            raise ValueError('Calibration/test image overlap')
        beta = frozen['beta']
    elif calibration is not None:
        raise ValueError('Replacement has no beta calibration')
    result = dict(**compute(meta, arrays, rule, beta), units='percent; gain in percentage points',
                  image_ids=meta['image_ids'], identity=meta['identity'],
                  export_sha256=digest(Path(root)/'export.json'),
                  replay_source_sha256=digest(__file__),
                  calibration_sha256=digest(calibration) if calibration else None)
    write_new(output, result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    command = commands.add_parser('export')
    for name in ('config', 'head-dir', 'feedback', 'output'):
        command.add_argument('--'+name, required=True)
    command.add_argument('--split', choices=['validation', 'test'], default='test')
    command.add_argument('--nec', type=int, default=5)
    command.add_argument('--budget', type=int, default=5)
    command = commands.add_parser('calibrate')
    command.add_argument('--export', dest='root', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--betas', type=float, nargs='+', required=True)
    command = commands.add_parser('evaluate')
    command.add_argument('--export', dest='root', required=True)
    command.add_argument('--output', required=True)
    command.add_argument('--rule', choices=['replace', 'additive'], default='replace')
    command.add_argument('--calibration')
    args = vars(parser.parse_args())
    action = args.pop('command')
    if action == 'export':
        from spatial_cbm.training.config import load
        args['cfg'] = load(args.pop('config'))
    result = globals()[action](**args)
    print(json.dumps({k: v for k, v in result.items() if k not in {'predictions_before', 'predictions_after', 'files', 'pairs', 'identity', 'input_sha256', 'image_ids'}}, indent=2))


if __name__ == '__main__':
    main()
