import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import pytest
import torch

from cub70.__main__ import evaluate as cub_evaluate, assignments, select_maps
from spatial_cbm.training.classifier import evaluate as classify
from spatial_cbm.training.data import prepare, verify, Samples, digest, human_targets, target_mask
from spatial_cbm.training.engine import fit
from spatial_cbm.training.export import export
from spatial_cbm.training.manifest import create
from spatial_cbm.training.metrics import assess
from test_training import inputs, config


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_classification_reports_nec5_and_explicit_average_without_concept_test_labels(tmp_path):
    prepared,_=inputs(tmp_path)
    bank=tmp_path/'six.txt';bank.write_text('head\ntail\na\nb\nc\nd\n')
    test=tmp_path/'test.jsonl'
    rows=[json.loads(line) for line in test.read_text().splitlines()]
    for row in rows: row.pop('detections')
    test.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    other=tmp_path/'six-prepared'
    prepare(tmp_path/'source.jsonl',other,'cub',bank,split=prepared/'split.json',heldout_source=test)
    cfg=config(tmp_path,other);fit(cfg);export(cfg)
    run=Path(cfg.training.output);heads=run/'nec';heads.mkdir()
    files={}
    for nec in (5,6):
        w=torch.zeros(2,6);w[:,:nec]=.001
        for prefix,tensor in [('W_g',w),('b_g',torch.tensor([2.,0.]) if nec==5 else torch.tensor([0.,2.]))]:
            name=f'{prefix}@NEC={nec}.pt';torch.save(tensor,heads/name);files[name]=digest(heads/name)
    identity=dict(checkpoint_sha256=digest(run/'best.pt'),
        feature_manifest_sha256=digest(run/'features_complete.json'),
        normalization_sha256=digest(run/'final_layer_normalization.pt'),head_files=files,nec_values=[5,6])
    (heads/'classifier_identity.json').write_text(json.dumps(identity))
    result=classify(cfg,heads,[5,6])
    assert result['images']==2 and result['ACC_NEC5']==50 and result['AVGACC']==50
    assert result['accuracy_by_nec']=={'5':50.,'6':50.}
    assert classify(cfg,heads,[5,6])==result
    torch.save(torch.zeros(2,6),heads/'W_g@NEC=5.pt')
    with pytest.raises(ValueError,match='changed NEC'): classify(cfg,heads,[5,6])


def test_partimagenet_human_supervision_uses_same_training_pipeline(tmp_path):
    prepared,_=inputs(tmp_path,'partimagenetpp')
    for name in ('source.jsonl','test.jsonl'):
        path=tmp_path/name;rows=[json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row['human_parts']=[dict(concept='head',polygons=[[.2,.2,.8,.2,.8,.8,.2,.8]],coordinates='normalized_xy')]
            row.pop('detections')
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    human=tmp_path/'human'
    prepare(tmp_path/'source.jsonl',human,'partimagenetpp',tmp_path/'concepts.txt',
        split=prepared/'split.json',heldout_source=tmp_path/'test.jsonl',supervision='human')
    assert verify(human)['training_images_sha256']==verify(prepared)['training_images_sha256']
    cfg=config(tmp_path,prepared,'partimagenetpp')
    cfg.training.stage='backbone';cfg.model.checkpoint='';fit(cfg)
    backbone=str(Path(cfg.training.output)/'best.pt')
    cfg.data.prepared=str(human);cfg.data.supervision='human';cfg.data.confidence=1.
    cfg.training.stage='concept';cfg.training.output=str(tmp_path/'human-run');cfg.model.checkpoint=backbone
    dataset=Samples(cfg,'train',verify(human),metrics=True)
    assert dataset[0][1].tolist()==[1.,0.]
    assert dataset[0][3].max()>0 and dataset[0][4].shape[-2:]==(32,32)
    fit(cfg);calibration=assess(cfg);test=assess(cfg,test=True)
    assert calibration['supervision'].startswith('human')
    assert test['metrics']['concept']['threshold']==calibration['metrics']['concept']['threshold']
    assert 'human regions' in test['metrics']['spatial_ap_definition']
    cfg.dataset='cub'
    with pytest.raises(ValueError,match='only for PartImageNet'): cfg.validate()


def test_human_targets_union_before_downsampling_and_no_detector_fallback():
    row=dict(human_parts=[dict(concept='part',coordinates='pixel_xy',polygons=[[0,0,1,0,1,7,0,7]]),
                          dict(concept='part',coordinates='pixel_xy',polygons=[[4,0,5,0,5,7,4,7]])])
    original=json.dumps(row)
    targets=human_targets(row)
    assert json.dumps(row)==original
    assert len(targets)==1
    assert target_mask(targets[0],(8,8),8,8,1)[0,0]==pytest.approx(.5)
    with pytest.raises(ValueError,match='never detector'): human_targets(dict(detections=[]))


def test_coco_human_adapter_joins_by_identity_and_requires_category_map(tmp_path):
    manifest=tmp_path/'images.jsonl'
    manifest.write_text(json.dumps(dict(id='bird.jpg',image='bird.jpg',class_id=0))+'\n')
    coco=tmp_path/'parts.json';coco.write_text(json.dumps(dict(
        images=[dict(id=17,file_name='bird.jpg')],annotations=[dict(image_id=17,category_id=8,
        segmentation=[[0,0,5,0,5,5,0,5]])])))
    mapping=tmp_path/'mapping.json';mapping.write_text(json.dumps({'8':'head'}))
    output=tmp_path/'human.jsonl'
    create('manifest',tmp_path,output,manifest,human_annotations=coco,category_map=mapping)
    assert json.loads(output.read_text())['human_parts'][0]['concept']=='head'
    mapping.write_text('{}')
    with pytest.raises(ValueError,match='cover every'): create('manifest',tmp_path,tmp_path/'bad.jsonl',manifest,human_annotations=coco,category_map=mapping)


def test_cub_selection_uses_only_mapped_fused_scores():
    mapping=dict(mappings=[dict(concept='head one',parts=['head']),dict(concept='head two',parts=['head'])])
    cols=assignments(mapping,['head one','unrelated','head two'])
    output=dict(final_logits=torch.tensor([[-2.,100.,-1.]]),spatial_maps=torch.arange(12.).reshape(1,3,2,2))
    _,selected=select_maps(output,cols,4)
    assert selected['head'].item()==2
    with pytest.raises(ValueError,match='absent'): assignments(mapping,['head one'])


def test_cub70_uses_heldout_roster_and_frozen_threshold(tmp_path):
    prepared,_=inputs(tmp_path);cfg=config(tmp_path,prepared);fit(cfg);calibration=assess(cfg)
    mapping=tmp_path/'parts.json';mapping.write_text(json.dumps(dict(mappings=[dict(concept='head',parts=['head'])])))
    mask_root=tmp_path/'masks'
    for i in range(2):
        folder=mask_root/str(i+1);folder.mkdir(parents=True)
        mask=Image.new('L',(48,40),0);ImageDraw.Draw(mask).rectangle((10,8,35,30),fill=255)
        mask.save(folder/f'test-{i}_head.png')
    result=cub_evaluate(cfg,mapping,tmp_path/'cub70.json',masks_root=mask_root)
    assert result['images']==2 and result['per_part']['head']['pairs']==2
    assert result['threshold']==calibration['metrics']['localization']['threshold']
    assert all(row['concept']=='head' for row in result['selections'])
    assert result['supported_parts']==1 and result['expected_parts']==8
    manifest=tmp_path/'wrong.jsonl';manifest.write_text(json.dumps(dict(id='0',masks={}))+'\n')
    with pytest.raises(ValueError,match='held-out'): cub_evaluate(cfg,mapping,tmp_path/'invalid.json',manifest=manifest)
