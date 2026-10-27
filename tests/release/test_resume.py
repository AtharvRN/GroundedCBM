import copy
from pathlib import Path

import pytest
import torch

import gcbm.training.engine as engine
from test_training import inputs, config


@pytest.mark.parametrize('backbone_ema', [False, True])
def test_interrupted_run_matches_uninterrupted_training(tmp_path,monkeypatch,backbone_ema):
    old_threads=torch.get_num_threads();torch.set_num_threads(1)
    try:
        prepared,_=inputs(tmp_path)
        cfg=config(tmp_path,prepared)
        if backbone_ema:
            cfg.training.stage='backbone'; cfg.model.checkpoint=''
            cfg.training.ema_decay=.9999
            cfg.training.mixup_alpha=.2; cfg.training.cutmix_alpha=1.
            cfg.training.randaugment_ops=2; cfg.training.random_erasing=.25
        engine.fit(cfg)
        expected=torch.load(Path(cfg.training.output)/'last.pt',weights_only=False)
        interrupted=copy.deepcopy(cfg);interrupted.training.output=str(tmp_path/'interrupted')
        original=engine.run_epoch
        calls=0
        def fail_after_first_epoch(*args,**kwargs):
            nonlocal calls
            calls+=1
            if calls==3: raise RuntimeError('simulated interruption')
            return original(*args,**kwargs)
        monkeypatch.setattr(engine,'run_epoch',fail_after_first_epoch)
        with pytest.raises(RuntimeError,match='simulated'):
            engine.fit(interrupted)
        monkeypatch.setattr(engine,'run_epoch',original)
        engine.fit(interrupted,resume=True)
        actual=torch.load(Path(interrupted.training.output)/'last.pt',weights_only=False)
        for group in ('model','ema_model') if backbone_ema else ('model',):
            for key in expected[group]:
                torch.testing.assert_close(expected[group][key],actual[group][key],rtol=0,atol=0)
        if backbone_ema:
            assert actual['ema_updates']==expected['ema_updates']==6
            best=torch.load(Path(interrupted.training.output)/'best.pt',weights_only=False)
            assert best['epoch']==min(actual['history'],key=lambda r:r['val_loss'])['epoch']
    finally:
        torch.set_num_threads(old_threads)
