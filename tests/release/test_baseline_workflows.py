"""Shared workflow smoke tests, with synthetic CLIP targets and real ResNet/solver."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from spatial_cbm.training import baselines, classifier, engine, teacher
from spatial_cbm.training.data import prepare, Samples
from spatial_cbm.training.config import load
from spatial_cbm.training.engine import fit
from spatial_cbm.training.export import export
from spatial_cbm.training.metrics import assess
from spatial_cbm.training.model import load_selected
from test_training import inputs, config


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def fixture(tmp_path, monkeypatch, method, dataset='cub', hidden=0, annotated=True):
    original,_ = inputs(tmp_path,dataset)
    names = ['head','tail','wing','bill','eye','leg']
    bank = tmp_path/'bank.txt'
    bank.write_text('\n'.join(names)+'\n')
    for name in ('source.jsonl','test.jsonl'):
        path = tmp_path/name
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            detection = row.pop('detections')[0]
            if annotated: row['detections'] = [dict(detection,concept=c) for c in names]
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    split = tmp_path/'split.json'
    split.write_text(json.dumps(dict(train=['0','2','4','5','7'],validation=['1','3','6'])))
    prepared = tmp_path/'baseline-prepared'
    prepare(tmp_path/'source.jsonl',prepared,dataset,bank,split=split,heldout_source=tmp_path/'test.jsonl')
    cfg = config(tmp_path,prepared,dataset)
    cfg.model.method, cfg.model.vlg_hidden = method,hidden
    cfg.model.salf_grid = 3
    cfg.training.batch_size = 3
    cfg.training.epochs = 1
    cfg.training.kl = cfg.training.spatial_bce = 0.
    cfg.training.lf_clip_cutoff = cfg.training.lf_interpretability_cutoff = -1.
    cfg.teacher.checkpoint = str(tmp_path/'clip.pt')
    Path(cfg.teacher.checkpoint).write_bytes(b'offline fixture')
    cfg.teacher.cache = str(tmp_path/'teacher')

    class Targets:
        def __init__(self, config, protocol, verified_manifest=None):
            self.cfg = config
        def get(self, position):
            # Nonlinear image-dependent cosine-like values; no test targets exist.
            assert 0<=position<8
            values = torch.sin(torch.arange(6)*.7+position*.8)*.5
            return (values[:,None,None].expand(-1,3,3).clone()
                    if self.cfg.model.method=='salf_cbm' else values)
    monkeypatch.setattr(teacher,'TeacherTargets',Targets)
    monkeypatch.setattr(teacher,'verify_teacher',lambda cfg,data: dict(synthetic=True))
    return cfg


@pytest.mark.parametrize('method', ['vlg_cbm','lf_cbm','salf_cbm'])
@pytest.mark.parametrize('dataset', ['cub','imagenet','places365','partimagenetpp'])
def test_baselines_share_training_calibration_export_and_nec(tmp_path,monkeypatch,method,dataset):
    cfg = fixture(tmp_path,monkeypatch,method,dataset)
    result = fit(cfg)
    assert result['best_epoch']==1
    assert fit(cfg,resume=True)==result
    backbone,head,bank,protocol,_ = load_selected(cfg)
    history = json.loads((Path(cfg.training.output)/'history.json').read_text())
    assert history[0]['train']['images']==5
    assert history[0]['validation']['images']==3
    assert np.isfinite(history[0]['val_loss'])
    assert history[0]['validation']['concept_ap'] is not None
    assert history[0]['validation']['spatial_ap'] is not None
    calibration = assess(cfg)
    test = assess(cfg,test=True)
    assert test['thresholds']==calibration['thresholds']
    assert test['map_kind']==('native' if method=='salf_cbm' else 'linear_cam')
    if method in {'lf_cbm','salf_cbm'}: assert test['metrics']['loss'] is None
    export(cfg)
    root = Path(cfg.training.output)
    assert (root/'concepts.txt').read_text().splitlines()==bank
    features = np.load(root/'features/train_features.npy')
    assert features.shape==(5,len(bank))
    samples = Samples(cfg,'train',protocol,concepts=False)
    with torch.inference_mode():
        expected = head(backbone(torch.stack([samples[i][0] for i in range(len(samples))])))['final_logits']
    np.testing.assert_allclose(features,expected.numpy(),rtol=1e-4,atol=1e-5)
    torch.testing.assert_close(samples[0][0],samples[0][0])
    assert export(cfg)['files']
    # Run actual GLM-SAGA, not manufactured classifier weights.
    monkeypatch.setattr(sys,'argv',['classifier','--artifact_dir',str(root),'--output_dir',str(root/'nec'),
        '--device','cpu','--seed','42','--saga_workers','0','--saga_batch_size','5',
        '--n_iters','5','--max_glm_steps','5','--lam_max','.00001','--epsilon','.01',
        '--nec_values','1','--skip_train_eval','--cache_features_device','cpu_preserve_sampler'])
    classifier.main()
    classification = classifier.evaluate(cfg,root/'nec',[1])
    assert classification['images']==2
    assert 0<=classification['AVGACC']<=100


def test_nonlinear_vlg_does_not_invent_spatial_maps(tmp_path,monkeypatch):
    cfg = fixture(tmp_path,monkeypatch,'vlg_cbm',hidden=1)
    fit(cfg)
    calibration = assess(cfg)
    test = assess(cfg,test=True)
    assert test['map_kind']=='unavailable'
    assert calibration['thresholds']['spatial'] is None
    assert test['metrics']['spatial_ap'] is None
    assert test['metrics']['localization']['foreground_iou'] is None
    export(cfg)


@pytest.mark.parametrize('method',['lf_cbm','salf_cbm'])
def test_clip_training_and_export_do_not_require_detector_labels(tmp_path,monkeypatch,method):
    cfg = fixture(tmp_path,monkeypatch,method,annotated=False)
    fit(cfg)
    export(cfg)
    with pytest.raises(ValueError,match='explicit annotations'):
        assess(cfg)


def test_regression_rejects_degenerate_training_batch(tmp_path,monkeypatch):
    cfg = fixture(tmp_path,monkeypatch,'lf_cbm')
    cfg.training.batch_size = 2
    with pytest.raises(ValueError,match='three images'):
        cfg.validate()


@pytest.mark.parametrize('method',['lf_cbm','salf_cbm'])
def test_real_cache_proof_survives_training_and_checkpoint_reload(tmp_path,monkeypatch,method):
    real_targets,real_verify = teacher.TeacherTargets,teacher.verify_teacher
    cfg = fixture(tmp_path,monkeypatch,method)
    monkeypatch.setattr(teacher,'TeacherTargets',real_targets)
    monkeypatch.setattr(teacher,'verify_teacher',real_verify)
    cfg.data.input_size, cfg.data.resize_size = 224,256

    class Encoder(torch.nn.Module):
        visual = SimpleNamespace(input_resolution=224)
        def encode_text(self, tokens):
            return tokens.float()
        def encode_image(self, images):
            rgb = images.mean((2,3))
            return torch.stack((rgb[:,0],rgb[:,1],rgb[:,2],rgb[:,0].square()+1),dim=1)

    encoder = Encoder()
    original_import = teacher.importlib.import_module
    def clip_import(name,*args,**kwargs):
        if name!='clip': return original_import(name,*args,**kwargs)
        from torchvision import transforms
        transform = transforms.Compose([transforms.Resize(224),transforms.CenterCrop(224),transforms.ToTensor()])
        return SimpleNamespace(load=lambda *a,**k:(encoder,transform),
            tokenize=lambda words: torch.tensor([[1.,i*.5,-.3,1.] for i in range(len(words))]))
    monkeypatch.setattr(teacher.importlib,'import_module',clip_import)
    manifest = teacher.prepare_teacher(cfg)
    assert manifest['rows_written']==8
    result = fit(cfg)
    assert fit(cfg,resume=True)==result
    export(cfg)
    assert assess(cfg)['identity']['teacher']==manifest
    assert assess(cfg,test=True)['metrics']['loss'] is None


@pytest.mark.parametrize('method',['lf_cbm','salf_cbm','vlg_cbm'])
def test_baseline_recipe_loads(method):
    root = Path(__file__).resolve().parents[2]
    cfg = load(root/f'configs/release/baselines/{method}.yaml')
    assert cfg.model.method==method
    assert cfg.training.kl==cfg.training.spatial_bce==0


def test_lf_post_filter_removal_preserves_resume_and_column_identity(tmp_path,monkeypatch):
    cfg = fixture(tmp_path,monkeypatch,'lf_cbm')
    cfg.training.lf_interpretability_cutoff = .45
    finalize = engine.finalize_lf
    # Inject known final-selection similarities, leaving training loss untouched.
    def controlled_filter(*args,**kwargs):
        with monkeypatch.context() as patch:
            patch.setattr(baselines.RegressionMeter,'similarities',
                lambda self: torch.tensor([.8,.45,-.2,.9,0.,.7]))
            return finalize(*args,**kwargs)
    monkeypatch.setattr(engine,'finalize_lf',controlled_filter)
    result = fit(cfg)
    root = Path(cfg.training.output)
    receipt = json.loads((root/'concept_filter.json').read_text())
    assert receipt['before']==list(range(6))
    assert receipt['retained']==[0,3,5]
    assert torch.load(root/'last.pt',weights_only=False)['model']['projection.weight'].shape[0]==6
    assert fit(cfg,resume=True)==result
    _,head,bank,_,_ = load_selected(cfg)
    assert head.concept_indices.tolist()==[0,3,5]
    assert bank==['head','bill','leg']
    export(cfg)
    assert np.load(root/'features/train_features.npy').shape==(5,3)
    assert (root/'concepts.txt').read_text().splitlines()==bank
    assert assess(cfg)['metrics']['concept_ap_total']==3
    assert assess(cfg,test=True)['metrics']['concept_ap_total']==3


def test_lf_fixed_vocabulary_survives_training_export_and_assessment(tmp_path, monkeypatch):
    cfg = fixture(tmp_path, monkeypatch, 'lf_cbm')
    cfg.training.filter_concepts = False
    # These cutoffs would reject every concept if either filter still ran.
    cfg.training.lf_clip_cutoff = cfg.training.lf_interpretability_cutoff = 1.
    result = fit(cfg)
    assert fit(cfg, resume=True) == result
    root = Path(cfg.training.output)
    _, head, bank, _, _ = load_selected(cfg)
    assert head.concept_indices.tolist() == list(range(6))
    assert bank == ['head', 'tail', 'wing', 'bill', 'eye', 'leg']
    assert not (root/'concept_filter.json').exists()
    export(cfg)
    assert np.load(root/'features/train_features.npy').shape == (5, 6)
    assert (root/'concepts.txt').read_text().splitlines() == bank
    assert assess(cfg)['metrics']['concept_ap_total'] == 6
    assert assess(cfg, test=True)['metrics']['concept_ap_total'] == 6


@pytest.mark.parametrize('method',['lf_cbm','salf_cbm'])
def test_cached_full_batch_projection_and_separate_inference_batch(tmp_path,monkeypatch,method):
    cfg = fixture(tmp_path,monkeypatch,method)
    cfg.training.cache_regression_features = True
    cfg.training.feature_batch_size = cfg.training.evaluation_batch_size = 2
    cfg.training.batch_size = 50000
    result = fit(cfg)
    assert fit(cfg,resume=True)==result
    backbone,head,_,data,_ = load_selected(cfg)
    protocol = engine.provenance(cfg,data)
    paths = baselines.prepare_regression_features(cfg,backbone,protocol)
    for split,pair in paths.items():
        samples = Samples(cfg,split,data,concepts=False)
        array = np.load(pair[0])
        with torch.inference_mode():
            expected = backbone(torch.stack([samples[i][0] for i in range(len(samples))]))['conv5']
            if method=='lf_cbm' and split=='train': expected = expected.mean((2,3),keepdim=True)
        np.testing.assert_allclose(array,expected.numpy(),rtol=1e-4,atol=1e-5)
    export(cfg)
    assess(cfg)
    assess(cfg,test=True)
    # Training/validation cache use bypasses the backbone, but export/evaluation do not.
    train,val = engine.loaders(cfg,data,teacher_manifest=protocol['teacher'],cached_features=paths)
    monkeypatch.setattr(backbone,'forward',lambda *_:pytest.fail('Cached training invoked backbone'))
    assert len(train)==1 and len(val)==2
    batch = next(iter(train))
    assert batch[0].shape[0]==5
    assert engine.run_epoch(backbone,head,val,cfg)['images']==3
    with open(paths['train'][0],'ab') as stream: stream.write(b'changed')
    with pytest.raises(ValueError,match='cache content changed'):
        baselines.prepare_regression_features(cfg,backbone,protocol)
