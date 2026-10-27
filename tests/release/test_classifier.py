import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from spatial_cbm.training.classifier import select_path_points_for_nec, save_thresholded_nec_weights
from spatial_cbm.training.data import digest


def test_exact_nec_ties_and_unreached_path(tmp_path):
    params=dict(weight=torch.ones(2,3),bias=torch.zeros(2),lam=.1,lr=.1,metrics={})
    selected=select_path_points_for_nec([params],3,[1])
    result=save_thresholded_nec_weights([params],selected,tmp_path)
    assert result[0]['truncated_nnz']==2
    with pytest.raises(ValueError,match='did not reach'):
        select_path_points_for_nec([dict(params,weight=torch.zeros(2,3))],3,[1])


def test_cpu_cached_solver_end_to_end(tmp_path):
    rng=np.random.default_rng(42)
    folder=tmp_path/'features';folder.mkdir()
    for split,n in [('train',32),('val',8)]:
        y=np.arange(n)%2
        x=rng.normal(0,.1,(n,3)).astype('float32')+(2*y-1)[:,None]
        np.save(folder/f'{split}_features.npy',x.astype('float32'))
        np.save(folder/f'{split}_targets.npy',y)
    torch.save(dict(mean=torch.zeros(3),std=torch.ones(3)),tmp_path/'final_layer_normalization.pt')
    torch.save({},tmp_path/'best.pt')
    files={str(p.relative_to(tmp_path)):digest(p) for p in folder.iterdir()}
    files['final_layer_normalization.pt']=digest(tmp_path/'final_layer_normalization.pt')
    (tmp_path/'features_complete.json').write_text(json.dumps(dict(files=files,identity=dict(checkpoint=digest(tmp_path/'best.pt')))))
    run=subprocess.run([sys.executable,'-m','spatial_cbm.training.classifier','--artifact_dir',str(tmp_path),
        '--output_dir',str(tmp_path/'nec'),'--device','cpu','--seed','42','--saga_workers','0',
        '--saga_batch_size','8','--n_iters','3','--max_glm_steps','5','--lam_max','.001',
        '--epsilon','.1','--nec_values','1','--skip_train_eval',
        '--cache_features_device','cpu_preserve_sampler'],capture_output=True,text=True)
    assert run.returncode==0,run.stdout+run.stderr
    weight=torch.load(tmp_path/'nec/W_g@NEC=1.pt',weights_only=True)
    assert int((weight.abs()>1e-5).sum())==2
