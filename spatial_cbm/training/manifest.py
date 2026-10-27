"""Small native-layout adapters; emitted records feed the same preparation code."""
import json
from pathlib import Path


def rows(layout, root, filelist=None, partition='train'):
    root = Path(root).resolve()
    if layout == 'imagefolder':
        classes = sorted(p for p in root.iterdir() if p.is_dir())
        for label, folder in enumerate(classes):
            for image in sorted(folder.rglob('*')):
                if image.suffix.lower() in {'.jpg','.jpeg','.png'}:
                    yield dict(id=str(image.relative_to(root)), image=str(image), class_id=label)
    elif layout == 'cub':
        names = dict(line.split(maxsplit=1) for line in (root/'images.txt').read_text().splitlines())
        labels = dict(line.split() for line in (root/'image_class_labels.txt').read_text().splitlines())
        for number, training in (line.split() for line in (root/'train_test_split.txt').read_text().splitlines()):
            if training == ('1' if partition=='train' else '0'):
                yield dict(id=names[number], image=str(root/'images'/names[number]), class_id=int(labels[number])-1)
    elif layout == 'places365':
        if not filelist:
            raise ValueError('Places365 requires the training file list')
        for line in Path(filelist).read_text().splitlines():
            name, category = line.rsplit(maxsplit=1)
            yield dict(id=name.lstrip('/'), image=str(root/name.lstrip('/')), class_id=int(category))
    elif layout == 'manifest':
        if not filelist:
            raise ValueError('PartImageNet++/custom input requires a JSONL manifest')
        with Path(filelist).open() as f:
            for line in f:
                row = json.loads(line)
                name = row.get('id', row.get('file_name'))
                path = row.get('image', row.get('path', row.get('file_name')))
                if not name or path is None:
                    raise ValueError('Manifest requires id/file_name and image/path/file_name')
                if isinstance(path, str):
                    path = str((root/Path(path)).resolve())
                yield dict(row, id=str(name), image=path, class_id=row['class_id'])
    else:
        raise ValueError('Unknown layout')


def coco_parts(path, category_map):
    """Resolve COCO image/category IDs explicitly, without filename-position joins."""
    payload = json.loads(Path(path).read_text())
    categories = json.loads(Path(category_map).read_text())
    images, names = {}, set()
    for image in payload['images']:
        if image['id'] in images or image['file_name'] in names:
            raise ValueError('Duplicate COCO image identity')
        images[image['id']] = image['file_name']; names.add(image['file_name'])
    result = {name:[] for name in names}
    for item in payload['annotations']:
        if item['image_id'] not in images: raise ValueError('Unknown COCO image_id')
        category = str(item['category_id'])
        if category not in categories: raise ValueError('Explicit category map must cover every annotation category')
        concept = categories[category]
        if concept is None: continue
        if not isinstance(concept,str) or not concept: raise ValueError('Invalid category-to-concept map')
        segmentation = item.get('segmentation')
        if isinstance(segmentation,dict): raise ValueError('RLE masks are unsupported; provide polygon annotations')
        if segmentation:
            target = dict(concept=concept,polygons=segmentation,coordinates='pixel_xy')
        elif 'bbox' in item:
            x,y,w,h = item['bbox']
            target = dict(concept=concept,box=[x,y,x+w,y+h],coordinates='pixel_xyxy')
        else:
            raise ValueError('Human annotation has neither polygons nor bbox')
        result[images[item['image_id']]].append(target)
    return result


def create(layout, root, output, filelist=None, annotation_map=None, partition='train', human_annotations=None, category_map=None):
    if bool(human_annotations) != bool(category_map) or (human_annotations and annotation_map):
        raise ValueError('Human COCO input needs a category map and cannot be mixed with GDINO annotations')
    mapping = None if annotation_map is None else json.loads(Path(annotation_map).read_text())
    human = None if human_annotations is None else coco_parts(human_annotations,category_map)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        for row in rows(layout, root, filelist,partition):
            if human is not None:
                key = row.get('annotation_image_id',row['id'])
                if key not in human: raise ValueError(f'Image identity missing from human COCO annotations: {key}')
                row.pop('detections',None); row.pop('annotation',None)
                row['human_parts'] = human[key]
            if mapping is not None:
                # Values contain annotation path, exact header identity, and coordinate convention.
                metadata = mapping[row['id']]
                if set(metadata)-{'annotation','annotation_image_id','box_coordinates'} or 'annotation' not in metadata:
                    raise ValueError('Invalid annotation mapping; do not infer ids from sorted filenames')
                metadata = dict(metadata)
                metadata['annotation'] = str((Path(annotation_map).resolve().parent/metadata['annotation']).resolve())
                row.update(metadata)
            stream.write(json.dumps(row)+'\n')
