import io
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest
import torch

from spatial_cbm.training.config import load
from spatial_cbm.training.data import prepare, verify, Samples
from spatial_cbm.training.manifest import create
from spatial_cbm.training.engine import learning_rate
from test_training import inputs, config


def test_all_release_yaml_files_load():
    for path in Path('configs/release').glob('*.yaml'):
        cfg=load(path)
        assert cfg.training.patience==10
        assert cfg.training.stage in {'concept','backbone'}
        assert cfg.training.positive_weight==1.


def test_split_is_saved_reusable_and_disjoint(tmp_path):
    prepared,rows=inputs(tmp_path)
    second=tmp_path/'second'
    prepare(tmp_path/'source.jsonl',second,'cub',tmp_path/'concepts.txt',split=prepared/'split.json')
    assert (prepared/'split.json').read_bytes()==(second/'split.json').read_bytes()
    bad=tmp_path/'bad.json'
    bad.write_text(json.dumps(dict(train=['0','1'],validation=['1'])))
    with pytest.raises(ValueError,match='disjoint'):
        prepare(tmp_path/'source.jsonl',tmp_path/'invalid','cub',split=bad)


def test_missing_annotations_do_not_become_negative_labels(tmp_path):
    prepared,rows=inputs(tmp_path)
    for row in rows: row.pop('detections')
    source=tmp_path/'no-annotations.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    other=tmp_path/'no-annotations'
    prepare(source,other,'cub',tmp_path/'concepts.txt',fraction=.25)
    cfg=config(tmp_path,other)
    with pytest.raises(ValueError,match='explicit annotations'):
        Samples(cfg,'train',verify(other))
    assert len(Samples(cfg,'train',verify(other),concepts=False))==6


def test_archive_offsets_match_extracted_images(tmp_path):
    prepared,rows=inputs(tmp_path)
    archive=tmp_path/'images.bin'
    data=bytearray()
    for row in rows:
        payload=Path(row['image']).read_bytes()
        row['image']=dict(archive=str(archive),offset=len(data),size=len(payload))
        data.extend(payload)
    archive.write_bytes(data)
    source=tmp_path/'archive.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    other=tmp_path/'archive-prepared'
    prepare(source,other,'cub',tmp_path/'concepts.txt',split=prepared/'split.json')
    cfg=config(tmp_path,prepared)
    original=Samples(cfg,'train',verify(prepared))[0]
    cfg.data.prepared=str(other)
    archived=Samples(cfg,'train',verify(other))[0]
    for a,b in zip(original[:-1],archived[:-1]): assert torch.equal(a,b)
    archive.write_bytes(b'')
    with pytest.raises(ValueError,match='Short archive'): Samples(cfg,'train',verify(other))[0]


def test_cub_adapter_excludes_official_test(tmp_path):
    (tmp_path/'images.txt').write_text('1 a.jpg\n2 b.jpg\n')
    (tmp_path/'image_class_labels.txt').write_text('1 1\n2 2\n')
    (tmp_path/'train_test_split.txt').write_text('1 1\n2 0\n')
    output=tmp_path/'cub.jsonl'
    create('cub',tmp_path,output)
    rows=[json.loads(l) for l in output.read_text().splitlines()]
    assert len(rows)==1 and rows[0]['id']=='a.jpg' and rows[0]['class_id']==0


def test_places_imagefolder_and_manifest_adapters(tmp_path):
    folder=tmp_path/'class';folder.mkdir()
    Image.new('RGB',(32,32)).save(folder/'a.jpg')
    output=tmp_path/'folder.jsonl';create('imagefolder',tmp_path,output)
    assert json.loads(output.read_text())['id']=='class/a.jpg'
    filelist=tmp_path/'places.txt';filelist.write_text('/class/a.jpg 0\n')
    output2=tmp_path/'places.jsonl';create('places365',tmp_path,output2,filelist)
    assert json.loads(output2.read_text())['image']==str(folder/'a.jpg')
    native=tmp_path/'pinpp.jsonl';native.write_text(json.dumps(dict(file_name='class/a.jpg',class_id=0))+'\n')
    output3=tmp_path/'pinpp-out.jsonl';create('manifest',tmp_path,output3,native)
    assert json.loads(output3.read_text())['image']==str(folder/'a.jpg')


def test_cosine_schedule_and_warmup():
    from spatial_cbm.training.config import Training
    cfg=Training(lr=.1,min_lr=.001,schedule='cosine',epochs=20,warmup_epochs=2)
    assert learning_rate(cfg,1)==.05
    assert learning_rate(cfg,2)==.1
    assert learning_rate(cfg,20)==.001


def test_imagefolder_evaluation_reuses_training_classes(tmp_path):
    for part,names in [('train',['a','z']),('test',['z'])]:
        for name in names:
            folder=tmp_path/part/name;folder.mkdir(parents=True)
            for i in range(2): Image.new('RGB',(8,8)).save(folder/f'{part}-{i}.jpg')
    train=tmp_path/'train.jsonl';test=tmp_path/'test.jsonl'
    create('imagefolder',tmp_path/'train',train)
    mapping=train.with_suffix('.classes.json')
    assert json.loads(mapping.read_text())=={'a':0,'z':1}
    with pytest.raises(ValueError,match='requires.*class-map'):
        create('imagefolder',tmp_path/'test',test,partition='test')
    assert not test.exists()
    create('imagefolder',tmp_path/'test',test,partition='test',class_map=mapping)
    rows=[json.loads(line) for line in test.read_text().splitlines()]
    assert all(row['class_id']==1 and row['class_name']=='z' for row in rows)
    prepare(train,tmp_path/'prepared','cub',fraction=.5,heldout_source=test)
    # Even if a caller incorrectly generates the test manifest as training,
    # preparation must catch the independent renumbering.
    wrong=tmp_path/'wrong.jsonl'
    create('imagefolder',tmp_path/'test',wrong)
    with pytest.raises(ValueError,match='class-name mapping differs'):
        prepare(train,tmp_path/'invalid','cub',fraction=.5,heldout_source=wrong)
    unknown=tmp_path/'test'/'unseen';unknown.mkdir()
    with pytest.raises(ValueError,match='absent from.*class map'):
        create('imagefolder',tmp_path/'test',tmp_path/'unknown.jsonl',partition='test',class_map=mapping)


@pytest.mark.parametrize('mapping',[{'a':0,'z':0},{'a':1},{'a':True},[],{}])
def test_imagefolder_rejects_invalid_class_maps(tmp_path,mapping):
    folder=tmp_path/'images'/'a';folder.mkdir(parents=True)
    path=tmp_path/'classes.json';path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError,match='Class map'):
        create('imagefolder',folder.parent,tmp_path/'out.jsonl',class_map=path)


def test_cli_help_and_classifier_import():
    for module in ['spatial_cbm.training','spatial_cbm.training.classifier','cub70']:
        run=subprocess.run([sys.executable,'-m',module,'--help'],capture_output=True,text=True)
        assert run.returncode==0,run.stderr
