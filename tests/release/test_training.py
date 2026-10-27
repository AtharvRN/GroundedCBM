import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch
from torchvision.models import resnet18

from gcbm.training.config import Config, Data, Model, Training, load
from gcbm.training.data import prepare, verify, Samples, occupancy
from gcbm.training.engine import fit, objective
from gcbm.training.export import export
from gcbm.training.model import Backbone, ConceptHead
from gcbm.validation_selection import minimum_loss, stop_for_loss
from gcbm.losses import soft_align_kl_loss


@pytest.fixture(autouse=True)
def threads():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def inputs(tmp_path, name='cub', seed=42):
    rows = []
    for i in range(8):
        image = tmp_path/f'{i}.png'
        Image.new('RGB', (48,40), (i*20,100,50)).save(image)
        rows.append(dict(id=str(i), image=str(image), class_id=i//4,
            detections=[dict(concept='head', confidence=.9, box=[.2,.2,.8,.8], coordinates='normalized_xyxy')]))
    source = tmp_path/'source.jsonl'
    source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    bank = tmp_path/'concepts.txt'; bank.write_text('head\ntail\n')
    output = tmp_path/f'prepared-{name}'
    prepare(source, output, name, bank, fraction=.25, seed=seed)
    return output, rows


def config(tmp_path, prepared, dataset='cub'):
    checkpoint = tmp_path/'backbone.pt'
    torch.manual_seed(7)
    torch.save(resnet18(weights=None).state_dict(), checkpoint)
    return Config(dataset, Data(str(prepared),str(tmp_path/'cache'),32,40,.15,0),
        Model('resnet18',str(checkpoint),'none','conv345',8,.1),
        Training(output=str(tmp_path/'run'),device='cpu',epochs=2,min_epochs=1,patience=10,
                 batch_size=2,precision='fp32',channels_last=False))


@pytest.mark.parametrize('dataset', ['cub','imagenet','places365','partimagenetpp'])
def test_same_training_export_and_resume_for_every_dataset(tmp_path, dataset):
    prepared, _ = inputs(tmp_path,dataset)
    cfg = config(tmp_path,prepared,dataset)
    result = fit(cfg)
    root = Path(cfg.training.output)
    history = json.loads((root/'history.json').read_text())
    assert result['best_epoch'] == min(history,key=lambda r:r['val_loss'])['epoch']
    assert all(np.isfinite(r['val_loss']) for r in history)
    assert history[0]['train']['images'] == 6 and history[0]['validation']['images'] == 2
    state = torch.load(root/'best.pt',weights_only=False)
    assert state['val_loss'] == min(r['val_loss'] for r in history)
    assert not any(k.startswith('network.') for k in state['model'])
    assert fit(cfg,resume=True) == result
    with pytest.raises(FileExistsError): fit(cfg)
    saved = export(cfg)
    x = np.load(root/'features/train_features.npy')
    stats = torch.load(root/'final_layer_normalization.pt',weights_only=True)
    np.testing.assert_allclose(stats['mean'],x.mean(0,dtype=np.float64),rtol=1e-5)
    assert export(cfg) == saved
    cfg.training.lr = .002
    with pytest.raises(ValueError,match='Resume protocol'): fit(cfg,resume=True)
    cfg.training.lr = .0001
    with Path(cfg.model.checkpoint).open('ab') as stream:
        stream.write(b'changed')
    with pytest.raises(ValueError,match='provenance changed: backbone_sha256'):
        export(cfg)


def test_shared_split_for_backbone_and_concepts(tmp_path):
    prepared, _ = inputs(tmp_path)
    cfg = config(tmp_path,prepared)
    cfg.training.stage='backbone'; cfg.training.epochs=1
    cfg.model.checkpoint=''
    fit(cfg)
    checkpoint = str(Path(cfg.training.output)/'best.pt')
    cfg.training.stage='concept'; cfg.training.output=str(tmp_path/'concept-run')
    cfg.model.checkpoint=checkpoint
    fit(cfg)
    protocol = verify(prepared)
    protocol['files']['split.json']='wrong'
    with pytest.raises(ValueError,match='same data and internal split'): Backbone(cfg.model,2,protocol)


def test_minimum_loss_patience_ties_and_legacy_history():
    history=[dict(epoch=i+1,val_loss=v) for i,v in enumerate([.5,.4,.4,.6])]
    assert minimum_loss(history)['epoch']==2
    assert stop_for_loss(history,2,1)
    assert not stop_for_loss(history[:3],2,1)
    with pytest.raises(ValueError): minimum_loss([dict(epoch=1)])
    with pytest.raises(ValueError): minimum_loss([dict(epoch=1,val_loss=float('nan'))])


def test_loss_matches_reference_kl_and_patch_bce():
    torch.manual_seed(4)
    maps=torch.randn(2,3,4,4,requires_grad=True)
    logits=torch.randn(2,3,requires_grad=True)
    labels=torch.tensor([[1.,0.,0.],[0.,1.,0.]])
    ids=torch.tensor([[0],[1]]); masks=torch.rand(2,1,4,4); valid=torch.ones(2,1,dtype=torch.bool)
    batch=[None,labels,ids,masks,valid,None]
    cfg=Training(positive_weight=1.)
    actual=objective(dict(final_logits=logits,spatial_maps=maps),batch,cfg)
    kl=soft_align_kl_loss(maps,ids,masks,valid)
    patch=torch.nn.functional.binary_cross_entropy_with_logits(maps[torch.arange(2),ids[:,0]],masks[:,0])
    expected=torch.nn.functional.binary_cross_entropy_with_logits(logits,labels)+.25*(kl+patch)
    torch.testing.assert_close(actual,expected)
    a=torch.autograd.grad(actual,(maps,logits),retain_graph=True)
    b=torch.autograd.grad(expected,(maps,logits))
    for x,y in zip(a,b): torch.testing.assert_close(x,y)


def test_targets_cache_crop_and_confidence(tmp_path,monkeypatch):
    prepared, _ = inputs(tmp_path)
    cfg=config(tmp_path,prepared)
    dataset=Samples(cfg,'train',verify(prepared))
    first=dataset[0]
    assert first[1].tolist()==[1.,0.]
    assert first[3].shape==(1,4,4)
    monkeypatch.setattr('gcbm.training.data.occupancy',lambda *a:pytest.fail('cache miss'))
    second=dataset[0]
    for x,y in zip(first[:-1],second[:-1]): assert torch.equal(x,y)
    a=occupancy([.2,.2,.8,.8],'normalized_xyxy',(48,40),40,32,4)
    b=occupancy([9.6,8.,38.4,32.],'pixel_xyxy',(48,40),40,32,4)
    np.testing.assert_allclose(a,b)


def test_identity_and_modified_inputs_fail(tmp_path):
    prepared, rows=inputs(tmp_path)
    (prepared/'concepts.txt').write_text('changed\n')
    with pytest.raises(ValueError,match='changed'): verify(prepared)
    ann=tmp_path/'a.json';ann.write_text(json.dumps([{'file_name':'wrong'}]))
    rows[0].pop('detections'); rows[0]['annotation']=str(ann)
    source=tmp_path/'wrong.jsonl';source.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    with pytest.raises(ValueError,match='identity mismatch'): prepare(source,tmp_path/'bad','cub')


def test_strict_configuration(tmp_path):
    path=tmp_path/'bad.yaml';path.write_text('training:\n  early_stopp_patience: 10\n')
    with pytest.raises(ValueError,match='Unknown'): load(path)


def test_model_conv345_matches_explicit_reference():
    cfg=Model(width=8)
    head=ConceptHead(cfg,(2,4,8),3)
    feats=dict(conv3=torch.randn(2,2,8,8),conv4=torch.randn(2,4,4,4),conv5=torch.randn(2,8,2,2))
    out=head(feats)
    coarse=head.fuse_spatial_features(feats)
    fine=head.conv3_proj(feats['conv3'])+torch.nn.functional.interpolate(coarse,size=(8,8),mode='bilinear',align_corners=False)
    maps=head.spatial_layer(torch.relu(head.refine(torch.relu(fine))))
    spatial=torch.logsumexp(maps.flatten(2),-1)-np.log(64)
    glob=head.global_layer(feats['conv5'].mean((2,3)))
    torch.testing.assert_close(out['final_logits'],glob+.1*spatial)
