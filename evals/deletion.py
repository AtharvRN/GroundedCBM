"""Top-1 concept patch deletion using frozen release checkpoints and NEC heads."""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

import torch
from torch.nn import functional as F

from spatial_cbm.training.config import load
from spatial_cbm.training.data import Samples, digest, save_json
from spatial_cbm.training.model import load_selected
from .vlm_export import cohort


def classifier(cfg, head_dir, identity, classes, concepts, nec):
    root, head_dir = Path(cfg.training.output), Path(head_dir)
    metadata = head_dir/'classifier_identity.json'
    recorded = json.loads(metadata.read_text())
    norm_path = root/'final_layer_normalization.pt'
    if recorded['checkpoint_sha256'] != identity['checkpoint_sha256']:
        raise ValueError('Classifier belongs to another concept checkpoint')
    if recorded['feature_manifest_sha256'] != digest(root/'features_complete.json'):
        raise ValueError('Classifier feature provenance changed')
    if recorded['normalization_sha256'] != digest(norm_path):
        raise ValueError('Classifier normalization changed')
    files = [metadata, root/'best.pt', root/'features_complete.json', norm_path]
    if cfg.model.checkpoint:
        files.append(Path(cfg.model.checkpoint))
    norm = torch.load(norm_path, map_location=cfg.training.device, weights_only=True)
    mean, std = norm['mean'].float(), norm['std'].float()
    if mean.shape != (concepts,) or std.shape != mean.shape or not (std > 0).all():
        raise ValueError('Invalid training-only normalization')
    tensors = []
    for prefix in ('W_g', 'b_g'):
        name = f'{prefix}@NEC={nec}.pt'
        path = head_dir/name
        if recorded['head_files'].get(name) != digest(path):
            raise ValueError('Missing or changed NEC head')
        tensors.append(torch.load(path, map_location=cfg.training.device, weights_only=True).float())
        files.append(path)
    weight, bias = tensors
    if weight.shape != (classes, concepts) or bias.shape != (classes,):
        raise ValueError('NEC head dimensions differ from the trained model')
    if not all(torch.isfinite(v).all() for v in (mean, std, weight, bias)):
        raise ValueError('Nonfinite classifier or normalization')
    if int((weight.abs() > 1e-5).sum()) != nec*classes:
        raise ValueError('Exact NEC mismatch')
    return weight, bias, mean, std, {str(p): digest(p) for p in files}


def select_concepts(z, weight, predicted):
    contribution = z * weight[predicted]
    eligible = (z > 0) & (contribution > 0)
    columns = contribution.masked_fill(~eligible, -torch.inf).argmax(1)
    return columns, eligible.any(1)


def target_patches(maps, size, patch_size):
    if patch_size < 1 or size % patch_size:
        raise ValueError('Input size must be divisible by positive patch size')
    resized = F.interpolate(maps[:, None].float(), size=(size, size),
                            mode='bilinear', align_corners=False)
    return F.avg_pool2d(resized, patch_size, patch_size).flatten(1).argmax(1)


def random_patches(image_id, seed, windows, repeats):
    # Key by image identity so controls stay paired across methods and batch sizes.
    key = hashlib.sha256(f'{seed}:{image_id}'.encode()).digest()
    generator = torch.Generator().manual_seed(int.from_bytes(key[:8], 'big'))
    return torch.randint(windows, (repeats,), generator=generator).tolist()


def erase(images, patches, patch_size):
    modified = images.clone()
    width = images.shape[-1] // patch_size
    for image, patch in zip(modified, patches):
        y, x = divmod(int(patch), width)
        image[:, y*patch_size:(y+1)*patch_size, x*patch_size:(x+1)*patch_size] = 0
    return modified


def deletion_batch(images, image_ids, predict, weight, patch_size=32, repeats=3, seed=0):
    """predict returns standardized concepts, class logits and spatial maps."""
    if repeats < 1 or images.shape[-2] != images.shape[-1]:
        raise ValueError('Positive repeats and square inputs are required')
    size = images.shape[-1]
    if patch_size < 1 or size % patch_size:
        raise ValueError('Input size must be divisible by positive patch size')
    z, logits, maps = predict(images)
    if maps is None:
        raise ValueError('Model supplies no spatial maps; an explicit saliency adapter is required')
    if not all(torch.isfinite(v).all() for v in (z, logits, maps)):
        raise ValueError('Nonfinite model output')
    if maps.ndim != 4 or maps.shape[:2] != z.shape:
        raise ValueError('Spatial map and concept dimensions differ')
    predicted = logits.argmax(1)
    columns, eligible = select_concepts(z, weight, predicted)
    clean_prob = logits.softmax(1).gather(1, predicted[:, None]).squeeze(1)
    rows = [dict(image_id=name, predicted_class=int(predicted[i]), eligible=bool(eligible[i]),
                 clean_probability=float(clean_prob[i])) for i, name in enumerate(image_ids)]
    indices = eligible.nonzero().flatten()
    if not len(indices):
        return rows
    selected = columns[indices]
    targeted = target_patches(maps[indices, selected], size, patch_size).tolist()
    controls = [random_patches(image_ids[int(i)], seed, (size//patch_size)**2, repeats)
                for i in indices]
    patches = [targeted] + [[r[j] for r in controls] for j in range(repeats)]
    probabilities, activations = [], []
    for window_ids in patches:
        perturbed_z, perturbed_logits, _ = predict(erase(images[indices], window_ids, patch_size))
        if not torch.isfinite(perturbed_z).all() or not torch.isfinite(perturbed_logits).all():
            raise ValueError('Nonfinite perturbed output')
        probabilities.append(perturbed_logits.softmax(1).gather(1, predicted[indices, None]).squeeze(1))
        activations.append(perturbed_z.gather(1, selected[:, None]).squeeze(1))
    for j, index in enumerate(indices.tolist()):
        class_drops = [100*float(clean_prob[index]-p[j]) for p in probabilities]
        concept_drops = [float(z[index, columns[index]]-v[j]) for v in activations]
        rows[index].update(concept_index=int(columns[index]), clean_concept=float(z[index, columns[index]]),
            contribution=float(z[index, columns[index]]*weight[predicted[index], columns[index]]),
            targeted_patch=targeted[j], random_patches=controls[j],
            targeted_class_drop=class_drops[0], random_class_drops=class_drops[1:],
            random_class_drop=sum(class_drops[1:])/repeats,
            targeted_concept_drop=concept_drops[0], random_concept_drops=concept_drops[1:],
            random_concept_drop=sum(concept_drops[1:])/repeats)
    return rows


def summarize(rows):
    eligible = [r for r in rows if r['eligible']]
    metrics = {}
    for kind in ('class', 'concept'):
        for control in ('targeted', 'random'):
            key = f'{control}_{kind}_drop'
            metrics[key] = sum(r[key] for r in eligible)/len(eligible) if eligible else None
        metrics[f'{kind}_gap'] = (metrics[f'targeted_{kind}_drop']-metrics[f'random_{kind}_drop']
                                  if eligible else None)
    return dict(images=len(rows), eligible=len(eligible), excluded=len(rows)-len(eligible), **metrics)


def evaluate(cfg, head_dir, output, *, nec=5, count=None, image_ids=None,
             seed=0, batch_size=16, patch_size=32, repeats=3):
    if min(nec, batch_size, patch_size, repeats) < 1 or cfg.data.input_size % patch_size:
        raise ValueError('Positive budgets and an input size divisible by patch size are required')
    if count is not None and image_ids is not None:
        raise ValueError('Choose count or an explicit image roster, not both')
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    backbone, head, bank, data, identity = load_selected(cfg)
    map_kind = getattr(head, 'map_kind', 'native')
    if map_kind == 'unavailable':
        raise ValueError('Model supplies no spatial maps; an explicit saliency adapter is required')
    weight, bias, mean, std, files = classifier(cfg, head_dir, identity, data['classes'], len(bank), nec)
    backbone.float().eval(); head.float().eval()
    dataset = Samples(cfg, 'test', data, concepts=False)
    with sqlite3.connect(f'file:{cfg.data.prepared}/{dataset.database}?mode=ro', uri=True) as db:
        lookup = {position: (name, label) for position, name, label in
                  db.execute('SELECT position,id,class_id FROM samples')}
    records = [(i, *lookup[int(position)]) for i, position in enumerate(dataset.indices)]
    if len({r[1] for r in records}) != len(records):
        raise ValueError('Duplicate held-out image IDs')
    if image_ids is not None:
        requested = json.loads(Path(image_ids).read_text())
        if not isinstance(requested, list) or not all(isinstance(v, str) for v in requested):
            raise ValueError('Image roster must be a JSON list of held-out image IDs')
        if not requested or len(set(requested)) != len(requested):
            raise ValueError('Empty or duplicate image roster')
        by_id = {r[1]: r for r in records}
        if not set(requested) <= by_id.keys():
            raise ValueError('Image roster contains non-held-out IDs')
        chosen = [by_id[name] for name in requested]
    else:
        chosen = cohort(records, len(records) if count is None else count, seed)
    protocol = dict(model=identity, dataset=cfg.dataset, method=cfg.model.method,
        files=files, source_sha256=digest(__file__), map_kind=map_kind, nec=nec,
        backbone=dict(architecture=cfg.model.backbone, weights=cfg.model.weights,
                      checkpoint=cfg.model.checkpoint),
        concept_bank=bank, image_ids=[r[1] for r in chosen], seed=seed,
        sampling='Explicit held-out roster' if image_ids else 'Class-balanced hash-ranked round robin',
        concept_selection='Maximum positive contribution with positive standardized activation; ties by concept index',
        patch_size=patch_size, input_size=cfg.data.input_size, resize_size=cfg.data.resize_size,
        patch_selection='Highest mean bilinearly resized map; nonoverlapping grid; ties in row-major order',
        replacement='Zero in normalized input space (RGB ImageNet mean)',
        random_controls=repeats, random_rule='Uniform with replacement; SHA256(seed:image_id); target overlap allowed',
        precision='FP32; TF32 disabled', batch_size=batch_size,
        units=dict(class_drop='percentage points', concept_drop='training-standardized activation'))
    output.mkdir(parents=True, exist_ok=False)
    save_json(output/'protocol.json', protocol)

    def predict(images):
        if cfg.training.channels_last:
            images = images.contiguous(memory_format=torch.channels_last)
        prediction = head(backbone(images))
        z = (prediction['final_logits'].float()-mean)/std
        return z, F.linear(z, weight, bias), prediction['spatial_maps']

    previous = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rows = []
    try:
        with torch.inference_mode(), torch.autocast(torch.device(cfg.training.device).type, enabled=False), \
                (output/'rows.jsonl').open('x') as stream:
            for start in range(0, len(chosen), batch_size):
                chunk = chosen[start:start+batch_size]
                images = torch.stack([dataset[r[0]][0] for r in chunk]).to(cfg.training.device).float()
                batch = deletion_batch(images, [r[1] for r in chunk], predict, weight, patch_size, repeats, seed)
                for i, (row, (_, _, label)) in enumerate(zip(batch, chunk)):
                    row.update(class_id=label, correct=row['predicted_class']==label,
                               input_sha256=hashlib.sha256(images[i].cpu().contiguous().numpy().tobytes()).hexdigest())
                    if row['eligible']:
                        row['concept'] = bank[row['concept_index']]
                    stream.write(json.dumps(row, allow_nan=False)+'\n')
                stream.flush()
                rows.extend(batch)
                print(json.dumps(dict(completed=len(rows), total=len(chosen))), flush=True)
        if any(digest(path) != expected for path, expected in files.items()):
            raise ValueError('Checkpoint or classifier files changed during evaluation')
        result = dict(primary=summarize(rows), correct_only=summarize([r for r in rows if r['correct']]),
                      clean_accuracy=sum(r['correct'] for r in rows)/len(rows),
                      aggregation='Equal eligible-image means; random controls averaged within image; signed drops',
                      units=protocol['units'], protocol_sha256=digest(output/'protocol.json'),
                      rows_sha256=digest(output/'rows.jsonl'))
        save_json(output/'results.json', result)
        save_json(output/'complete.json', dict(results_sha256=digest(output/'results.json'), images=len(rows)))
        return result
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = previous


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--head-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--nec', type=int, default=5)
    sampling = parser.add_mutually_exclusive_group()
    sampling.add_argument('--count', type=int)
    sampling.add_argument('--image-ids', help='JSON list of held-out IDs; use the same list across models')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--patch-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = vars(parser.parse_args())
    cfg = load(args.pop('config'))
    print(json.dumps(evaluate(cfg, **args), indent=2))


if __name__ == '__main__':
    main()
