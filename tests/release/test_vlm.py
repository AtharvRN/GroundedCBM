import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from evals import vlm
from evals.vlm_export import cohort, export


@pytest.fixture(autouse=True)
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_export(path, split='validation', name='validation-image'):
    path.mkdir(); Image.new('RGB',(8,8),(100,150,200)).save(path/'clean.png')
    np.save(path/'native.npy',np.arange(4,dtype=np.float32).reshape(2,2))
    rows=[dict(id=name+':'+str(i),image_id=name,concept=concept,clean='clean.png',native='native.npy',
        clean_sha256=vlm.file_digest(path/'clean.png'),native_sha256=vlm.file_digest(path/'native.npy'))
        for i,concept in enumerate(('present part','absent part'))]
    manifest=path/'manifest.jsonl';manifest.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    vlm.save(path/'export.json',dict(identity=dict(dataset='cub',checkpoint_sha256='frozen'),
        split=split,image_ids=[name],images=1,pairs=2,manifest_sha256=vlm.file_digest(manifest)))
    return path


class FakeClient:
    model='alias'; expected='pinned-backend'
    def __init__(self):self.calls=[]
    def query(self,body):
        self.calls.append(body)
        present='absent part' not in body['messages'][1]['content'][0]['text']
        return dict(answer=dict(verdict='yes' if present else 'no',reason='visible evidence'),model=self.expected)


def arguments(command,folder,output,calibration=None):
    return SimpleNamespace(command=command,export=folder,output=output,thresholds=[.5,.1],workers=2,
        base_url='https://judge.example/v1',model='alias',expected_model='pinned-backend',
        api_key_env='UNUSED_TEST_KEY',attempts=1,timeout=1,calibration=calibration)


def test_calibration_and_two_fresh_repeats_resume_and_agreement(tmp_path):
    cal=make_export(tmp_path/'cal');test=make_export(tmp_path/'test','test','test-image')
    client=FakeClient(); out=tmp_path/'calibration'
    vlm.judge(arguments('calibrate',cal,out),client)
    assert vlm.read(out/'frozen_threshold.json')['threshold']==.1
    # Two presence calls reused across thresholds; one localization per threshold.
    assert len(client.calls)==4
    runs=[]
    for i in range(2):
        args=arguments('evaluate',test,tmp_path/f'repeat{i}',out/'frozen_threshold.json')
        other=FakeClient(); result=vlm.judge(args,other)
        assert result['concept_precision']==.5 and result['localization_given_presence']==1
        assert result['grounded_concept_precision']==.5 and len(other.calls)==3
        vlm.judge(args,other); assert len(other.calls)==3
        runs.append(args.output)
    summary=vlm.summarize(runs)
    assert summary['metrics']['grounded_concept_precision']==dict(mean=.5,sd=0.)
    assert summary['common_present_pairs']==1 and summary['presence_unanimous_agreement']==1
    assert summary['localization_unanimous_agreement_common_present']==1


def test_asset_tampering_and_external_paths_rejected(tmp_path):
    folder=make_export(tmp_path/'export');vlm.load_export(folder)
    (folder/'clean.png').write_bytes(b'changed')
    with pytest.raises(ValueError,match='asset changed'):vlm.load_export(folder)


def test_overlap_and_changed_frozen_threshold_rejected(tmp_path):
    cal=make_export(tmp_path/'cal');out=tmp_path/'calibration'
    vlm.judge(arguments('calibrate',cal,out),FakeClient())
    test=make_export(tmp_path/'test','test','validation-image')
    args=arguments('evaluate',test,tmp_path/'run',out/'frozen_threshold.json')
    with pytest.raises(ValueError,match='overlap'):vlm.judge(args,FakeClient())
    value=vlm.read(args.calibration);value['threshold']=.5;vlm.save(args.calibration,value)
    with pytest.raises(ValueError,match='threshold changed'):vlm.judge(args,FakeClient())


def test_failures_are_not_unsure_and_can_resume(tmp_path):
    folder=make_export(tmp_path/'export');args=arguments('calibrate',folder,tmp_path/'run')
    class Failing(FakeClient):
        def query(self,body):raise RuntimeError('transport failure')
    with pytest.raises(RuntimeError,match='Incomplete judgments'):vlm.judge(args,Failing())
    assert not (args.output/'complete.json').exists()
    assert list(args.output.glob('thresholds/*/failures.json'))
    assert vlm.judge(args,FakeClient())['concept_precision']==.5


def test_masks_prompts_and_unsure_denominators(tmp_path):
    assert not vlm.threshold_mask(np.ones((2,2)),.5,(8,8)).any()
    with pytest.raises(ValueError):vlm.threshold_mask(np.array([[np.nan]]),.5,(8,8))
    assert 'ImageNet photographs' in vlm.prompts('imagenet')[0]
    assert 'specific object or localized attribute' in vlm.prompts('places365')[1]
    value=vlm.metrics([dict(presence='yes',localization='unsure'),dict(presence='unsure',localization='not_assessed')])
    assert value['concept_precision']==.5 and value['localization_given_presence']==0


def test_client_requires_https_and_exact_backend(monkeypatch):
    monkeypatch.setenv('VLM_TEST_KEY','not-a-real-key')
    with pytest.raises(ValueError,match='HTTPS'):vlm.Client('http://judge.example','a','b','VLM_TEST_KEY')
    class Response:
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def read(self):return json.dumps(dict(model='wrong',choices=[])).encode()
    monkeypatch.setattr(vlm.urllib.request,'build_opener',lambda *_:SimpleNamespace(open=lambda *a,**k:Response()))
    client=vlm.Client('https://judge.example','a','b','VLM_TEST_KEY',attempts=1)
    with pytest.raises(RuntimeError,match='identity'):client.query({})


def test_cohort_selection_is_balanced_and_pre_inference():
    rows=[(i,f'image{i}',i//4) for i in range(12)]
    selected=cohort(rows,6,42)
    assert selected==cohort(list(reversed(rows)),6,42)
    assert all(sum(r[2]==label for r in selected)==2 for label in range(3))


def test_release_checkpoint_export_smoke(tmp_path):
    from test_training import inputs,config
    from spatial_cbm.training.engine import fit
    from spatial_cbm.training.export import export as features
    from spatial_cbm.training.data import digest
    prepared,_=inputs(tmp_path);cfg=config(tmp_path,prepared)
    fit(cfg);features(cfg)
    root=Path(cfg.training.output);heads=root/'nec';heads.mkdir()
    files={}
    for name,value in [('W_g@NEC=1.pt',torch.tensor([[1.,1.],[-1.,-1.]])),('b_g@NEC=1.pt',torch.zeros(2))]:
        torch.save(value,heads/name);files[name]=digest(heads/name)
    vlm.save(heads/'classifier_identity.json',dict(checkpoint_sha256=digest(root/'best.pt'),
        normalization_sha256=digest(root/'final_layer_normalization.pt'),head_files=files))
    from spatial_cbm.training import config as config_module
    from unittest.mock import patch
    args=SimpleNamespace(config=tmp_path/'config.yaml',head_dir=heads,split='test',count=2,seed=0,nec=1,top_k=2,output=tmp_path/'maps')
    with patch.object(config_module,'load',return_value=cfg):result=export(args)
    assert result['images']==2 and result['identity']['checkpoint_sha256']==digest(root/'best.pt')
    _,_,rows=vlm.load_export(args.output)
    assert rows and all(r['contribution']>0 for r in rows)
