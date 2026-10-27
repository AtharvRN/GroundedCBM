import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
import torch

from spatial_cbm.training.data import prepare, verify, Samples
from spatial_cbm.training.engine import fit
from spatial_cbm.training.metrics import Metrics, row_ap, assess
from test_training import inputs, config


def test_spatial_ap_handles_ties_like_sklearn():
    scores=torch.tensor([[1.,1.,0.,0.],[3.,2.,1.,0.],[0.,0.,0.,0.]])
    labels=torch.tensor([[1,0,0,1],[0,1,1,0],[1,0,0,0]],dtype=torch.bool)
    expected=[average_precision_score(y,s) for y,s in zip(labels.numpy(),scores.numpy())]
    np.testing.assert_allclose(row_ap(scores,labels).numpy(),expected,rtol=1e-12)


def metric_batch():
    labels=torch.tensor([[1.,0.,1.],[0.,1.,0.]])
    scores=torch.tensor([[.1,.2,1.],[-1.,.3,.8]])
    maps=torch.tensor([[[[1.,1.],[0.,0.]]]*3,[[[0.,2.],[1.,0.]]]*3])
    masks=torch.tensor([[[[1.,0.],[0.,0.]]],[[[0.,1.],[0.,0.]]]])
    batch=[None,labels,torch.tensor([[0],[1]]),masks,torch.ones((2,1),dtype=torch.bool),masks,None]
    return dict(final_logits=scores,spatial_maps=maps),batch


def test_calibration_maximizes_micro_f1_and_freezes_thresholds(tmp_path):
    output,batch=metric_batch()
    metrics=Metrics(2,3,tmp_path,calibrate=True)
    try:
        metrics.update(output,batch)
        result=metrics.result()
    finally:
        metrics.close()
    truth=batch[1].numpy().astype(bool).flatten()
    scores=output['final_logits'].sigmoid().numpy().flatten()
    grid=np.array(result['search']['concept_thresholds'])
    predicted=scores[:,None]>=grid[None,:]
    tp=(predicted & truth[:,None]).sum(0)
    f1=2*tp/(predicted.sum(0)+truth.sum())
    best=np.flatnonzero(f1==f1.max())[-1]
    assert result['concept']['threshold']==grid[best]
    assert result['concept']['f1']==pytest.approx(f1[best])
    expected=np.mean([average_precision_score(batch[1][:,j],output['final_logits'][:,j]) for j in range(3)])
    assert result['concept_ap']==pytest.approx(expected)
    thresholds=dict(concept=result['concept']['threshold'],spatial=result['localization']['threshold'])
    frozen=Metrics(2,3,tmp_path,thresholds=thresholds)
    try:
        output['final_logits'] *= -1
        frozen.update(output,batch)
        actual=frozen.result()
    finally:
        frozen.close()
    assert actual['concept']['threshold']==thresholds['concept']
    assert actual['localization']['threshold']==thresholds['spatial']
    assert 'search' not in actual


def test_vlg_split_matches_torch_permutation_and_rejects_test_overlap(tmp_path):
    prepared,_=inputs(tmp_path)
    split=json.loads((prepared/'split.json').read_text())
    order=torch.randperm(8,generator=torch.Generator().manual_seed(42)).tolist()
    assert split==dict(train=[str(i) for i in order[:6]],validation=[str(i) for i in order[6:]])
    assert verify(prepared)['split_method']=='vlg_randperm_truncated'
    with pytest.raises(ValueError,match='test overlaps'):
        prepare(tmp_path/'source.jsonl',tmp_path/'overlap','cub',tmp_path/'concepts.txt',
                fraction=.25,heldout_source=tmp_path/'source.jsonl')


@pytest.mark.parametrize('dataset,count',[('cub',10),('imagenet',10),('places365',5),('partimagenetpp',5)])
def test_dataset_holdout_defaults_use_training_pool_only(tmp_path,dataset,count):
    rows=[dict(id=str(i),image=f'image-{i}.jpg',class_id=0) for i in range(100)]
    source=tmp_path/'train.jsonl';source.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    test=tmp_path/'test.jsonl';test.write_text(json.dumps(dict(id='test',image='test.jpg',class_id=0))+'\n')
    result=prepare(source,tmp_path/'prepared',dataset,heldout_source=test)
    assert result['validation']==count and result['train']==100-count
    assert result['test_images']==1 and result['test_disjoint_verified']
    assert result['split_seed']==6885


def test_constant_maps_and_absent_concepts_are_explicit(tmp_path):
    output,batch=metric_batch()
    output['spatial_maps'].zero_()
    batch[1].zero_()
    metrics=Metrics(2,3,tmp_path,calibrate=True)
    try:
        metrics.update(output,batch)
        result=metrics.result()
    finally:
        metrics.close()
    assert result['concept_ap'] is None
    assert result['concept_ap_supported']==0
    assert result['concept']['threshold']>1
    assert result['concept']['predicted_positives']==0
    assert result['spatial_ap']==pytest.approx(.25)
    assert result['localization']['foreground_iou']==pytest.approx(.25)


def test_selected_checkpoint_calibration_and_test_are_separate(tmp_path,monkeypatch):
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        prepared,_=inputs(tmp_path)
        cfg=config(tmp_path,prepared)
        fit(cfg)
        with pytest.raises(FileNotFoundError): assess(cfg,test=True)
        seen=[]
        original=Samples.__getitem__
        def observed(self,index):
            seen.append(self.database)
            return original(self,index)
        monkeypatch.setattr(Samples,'__getitem__',observed)
        calibration=assess(cfg)
        assert set(seen)=={'samples.sqlite'}
        assert calibration['cohort']=='validation'
        assert calibration['metrics']['concept']['f1']>=0
        assert (Path(cfg.training.output)/'calibration.json').exists()
        seen.clear()
        evaluated=assess(cfg,test=True)
        assert set(seen)=={'test.sqlite'}
        assert evaluated['metrics']['concept']['threshold']==calibration['metrics']['concept']['threshold']
        assert 'search' not in evaluated['metrics']
        assert assess(cfg)==calibration
        assert assess(cfg,test=True)==evaluated
        calibration['identity']['checkpoint_sha256']='wrong'
        (Path(cfg.training.output)/'calibration.json').write_text(json.dumps(calibration))
        with pytest.raises(ValueError,match='Thresholds do not belong'): assess(cfg,test=True)
    finally:
        torch.set_num_threads(old_threads)
