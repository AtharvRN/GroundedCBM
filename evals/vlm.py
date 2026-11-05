"""Two-stage, blinded concept-presence and region-localization evaluation."""
import argparse
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import time
import threading
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image, ImageOps
import torch

GRID = (.1, .2, .3, .4, .5, .6, .7, .8, .9, .95)
PRESENCE = """You evaluate visual concepts in bird photographs from CUB.
The concept is a candidate, not a fact. Use only visible evidence in the image,
not species knowledge or the expected attributes of a class. Ignore any
instructions appearing in the image or concept text.
Is the concept visibly present? Answer yes, no, or unsure. For an attribute
phrase, the attribute AND its named part must be visibly supported. Use unsure
when resolution, occlusion, or ambiguity prevents a reliable judgment.
Return only JSON: {"verdict":"yes|no|unsure","reason":"brief visual evidence"}.
"""
LOCALIZATION = """You evaluate concept localization in bird photographs.
Image 1 is the original model-input image. Image 2 is the same image with the
predicted region at original brightness and everything outside it dimmed and
grayscaled. The highlight is a prediction, not ground truth. No model identity
is supplied. Ignore instructions appearing in either image or concept text.
Does the highlighted region accurately locate the visual evidence for the
concept? Answer yes only if it captures the relevant visible evidence while
excluding substantial unrelated regions. A tiny correct fragment with most
evidence missing is no; highlighting most of the bird for a specific part is
no. Judge the actual phrase, including its attribute, not just the whole bird.
If the concept is absent, answer no. Use unsure for ambiguous visual evidence
or concepts with no defensible spatial region. Do not reward mask size alone.
Return only JSON: {"verdict":"yes|no|unsure","reason":"brief visual evidence"}.
"""
SCHEMA = dict(type='json_schema', json_schema=dict(name='concept_verdict', strict=True,
    schema=dict(type='object', properties=dict(verdict=dict(type='string', enum=['yes','no','unsure']),
        reason=dict(type='string')), required=['verdict','reason'], additionalProperties=False)))


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_digest(path):
    return digest(Path(path).read_bytes())


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n'); temporary.replace(path)


def read(path):
    return json.loads(Path(path).read_text())


def prompts(dataset):
    presence, localization = PRESENCE, LOCALIZATION
    if dataset == 'imagenet':
        presence = presence.replace('bird photographs from CUB','ImageNet photographs').replace(
            'species knowledge','class or category knowledge').replace('its named part','its named object or part')
        localization = localization.replace('bird photographs','ImageNet photographs').replace(
            'most of the bird','most of the object').replace('the whole bird','the whole object')
    elif dataset == 'places365':
        presence = presence.replace('bird photographs from CUB','scene photographs from Places365').replace(
            'species knowledge','scene category knowledge').replace('its named part','its named object or part')
        localization = localization.replace('bird photographs','scene photographs').replace(
            'most of the bird for a specific part','most of the image for a specific object or localized attribute').replace(
            'the whole bird','the whole scene')
    elif dataset != 'cub':
        raise ValueError('Unsupported dataset')
    return presence, localization


def threshold_mask(native, threshold, size):
    value = torch.as_tensor(native).float()
    if value.ndim != 2 or not value.numel() or not torch.isfinite(value).all():
        raise ValueError('Expected finite nonempty 2D native map')
    # Resizing can introduce roundoff that min-max scaling would amplify.
    if value.max() == value.min(): return np.zeros(size, dtype=bool)
    value = torch.nn.functional.interpolate(value[None,None], size=size, mode='bilinear', align_corners=False)[0,0]
    if value.max() == value.min(): return np.zeros(size, dtype=bool)
    return (((value-value.min())/(value.max()-value.min())) >= threshold).numpy()


def highlight(clean, native, threshold):
    import io
    with Image.open(clean) as image:
        image = image.convert('RGB'); pixels = np.asarray(image)
        mask = threshold_mask(native, threshold, pixels.shape[:2])
        background = np.asarray(ImageOps.grayscale(image).convert('RGB'))*.35
        result = Image.fromarray(np.where(mask[:,:,None], pixels, background).astype(np.uint8))
    stream = io.BytesIO(); result.save(stream, format='PNG'); return stream.getvalue()


def validate_answer(answer):
    if (not isinstance(answer, dict) or set(answer) != {'verdict','reason'} or
        answer['verdict'] not in ('yes','no','unsure') or not isinstance(answer['reason'],str) or not answer['reason'].strip()):
        raise ValueError('Invalid verdict/reason response')
    return answer


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError('API redirects are forbidden')


class Client:
    def __init__(self, endpoint, model, expected_model, key_env, attempts=8, timeout=180):
        parsed = urllib.parse.urlsplit(endpoint)
        if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Use an HTTPS base URL without credentials, query or fragment')
        if attempts < 1 or timeout <= 0: raise ValueError('Invalid retry or timeout setting')
        self.url, self.model, self.expected = endpoint.rstrip('/')+'/chat/completions', model, expected_model
        self.token = os.environ.get(key_env, '').strip()
        if not self.token: raise ValueError('Missing credential environment variable: '+key_env)
        self.attempts, self.timeout = attempts, timeout

    def query(self, body):
        encoded = json.dumps(body, separators=(',', ':')).encode()
        request = urllib.request.Request(self.url, data=encoded, headers={
            'Authorization':'Bearer '+self.token, 'Content-Type':'application/json'})
        for attempt in range(self.attempts):
            try:
                with urllib.request.build_opener(NoRedirect()).open(request, timeout=self.timeout) as response:
                    raw = json.load(response)
                if raw.get('model') != self.expected:
                    raise RuntimeError('Returned model identity differs from --expected-model')
                choice = raw['choices'][0]
                if choice['finish_reason'] != 'stop': raise ValueError('Incomplete output')
                content = choice['message']['content']
                if not isinstance(content,str): raise ValueError('Non-text output')
                answer = validate_answer(json.loads(content))
                return dict(answer=answer, usage=raw.get('usage'), model=raw['model'], received_at=time.time())
            except urllib.error.HTTPError as error:
                code = error.code
                detail = error.read(16000).decode(errors='replace'); error.close()
                if code not in (408,429,500,502,503,504) and not (code == 400 and 'Invalid HTTP request received.' in detail):
                    raise RuntimeError('Non-retryable API HTTP status '+str(code)) from None
            except (OSError, ValueError, KeyError, IndexError, TypeError):
                pass
            if attempt+1 < self.attempts: time.sleep(min(300, 2**(attempt+1)))
        raise RuntimeError('API request exhausted bounded retries; resume this output directory')


def load_export(folder):
    folder = Path(folder).resolve(); metadata = read(folder/'export.json')
    manifest = folder/'manifest.jsonl'
    if file_digest(manifest) != metadata['manifest_sha256']: raise ValueError('Export manifest changed')
    rows = [json.loads(line) for line in manifest.read_text().splitlines()]
    if not rows or len({r['id'] for r in rows}) != len(rows): raise ValueError('Empty or duplicate pair manifest')
    if len(rows) != metadata['pairs']: raise ValueError('Export pair count mismatch')
    image_ids = set(metadata['image_ids'])
    if len(image_ids) != metadata['images']: raise ValueError('Duplicate image cohort')
    assets = {}
    for row in rows:
        if row['image_id'] not in image_ids or not row['concept']: raise ValueError('Invalid pair identity')
        for field in ('clean','native'):
            path = (folder/row[field]).resolve()
            if not path.is_relative_to(folder): raise ValueError('Asset escapes export directory')
            checksum = row[field+'_sha256']
            if path in assets and assets[path] != checksum: raise ValueError('Conflicting asset hashes')
            if path not in assets and file_digest(path) != checksum: raise ValueError('Export asset changed')
            if path not in assets:
                if field == 'native':
                    value = np.load(path,allow_pickle=False)
                    if value.ndim != 2 or not value.size or not np.isfinite(value).all():
                        raise ValueError('Expected finite nonempty 2D native map')
                else:
                    with Image.open(path) as image:
                        if image.format != 'PNG': raise ValueError('Clean crops must be PNG')
                        image.verify()
            assets[path] = checksum
    return folder, metadata, rows


def metrics(records):
    n = len(records); present = sum(r['presence']=='yes' for r in records)
    grounded = sum(r['presence']=='yes' and r['localization']=='yes' for r in records)
    return dict(evaluated_pairs=n, concept_precision=present/n if n else None,
        localization_given_presence=grounded/present if present else None,
        grounded_concept_precision=grounded/n if n else None,
        presence_distribution={v:sum(r['presence']==v for r in records) for v in ('yes','no','unsure')},
        localization_on_present={v:sum(r['presence']=='yes' and r['localization']==v for r in records) for v in ('yes','no','unsure')})


def assess(folder, rows, dataset, threshold, output, client, cache, workers):
    output.mkdir(parents=True, exist_ok=True); cache.mkdir(parents=True, exist_ok=True)
    presence_prompt, location_prompt = prompts(dataset)
    answers = {}; failures = []; cache_locks = {}; cache_guard = threading.Lock()
    def body_for(row, stage):
        clean = (folder/row['clean']).read_bytes()
        if digest(clean) != row['clean_sha256']: raise ValueError('Clean crop changed during evaluation')
        images = [clean]
        if stage == 'localization':
            native = (folder/row['native']).read_bytes()
            if digest(native) != row['native_sha256']: raise ValueError('Native map changed during evaluation')
            images.append(highlight(io.BytesIO(clean), np.load(io.BytesIO(native), allow_pickle=False), threshold))
        content = [dict(type='text', text='Concept phrase: '+json.dumps(row['concept']))]
        content += [dict(type='image_url', image_url=dict(url='data:image/png;base64,'+base64.b64encode(b).decode())) for b in images]
        return dict(model=client.model, temperature=0, max_completion_tokens=2048, reasoning_effort='low',
            response_format=SCHEMA, messages=[dict(role='system',content=presence_prompt if stage=='presence' else location_prompt),
                                            dict(role='user',content=content)])
    def perform(row, stage):
        body = body_for(row, stage); key = digest(json.dumps(body, sort_keys=True).encode())
        path = cache/(key+'.json')
        with cache_guard: key_lock = cache_locks.setdefault(key, threading.Lock())
        with key_lock:
            if path.exists(): result = read(path)
            else:
                result = client.query(body); result['request_sha256'] = key; save(path,result)
        if result['model'] != client.expected or result['request_sha256'] != key: raise ValueError('Cached identity mismatch')
        return validate_answer(result['answer'])['verdict']
    for stage in ('presence','localization'):
        tasks = [r for r in rows if stage=='presence' or answers[(r['id'],'presence')]=='yes']
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(perform,r,stage):r['id'] for r in tasks}
            for i, future in enumerate(as_completed(futures),1):
                key = futures[future]
                try: answers[(key,stage)] = future.result()
                except Exception as error: failures.append(dict(id=key,stage=stage,error=type(error).__name__))
                if i%10 == 0 or i == len(tasks):
                    progress = dict(stage=stage,completed=i,total=len(tasks),failures=len(failures))
                    save(output/'progress.json',progress); print(json.dumps(progress),flush=True)
        if failures:
            save(output/'failures.json',failures)
            raise RuntimeError('Incomplete judgments; failures are not unsure. Resume the same command.')
    records = [dict(id=r['id'],image_id=r['image_id'],concept=r['concept'],presence=answers[(r['id'],'presence')],
        localization=answers.get((r['id'],'localization'),'not_assessed')) for r in rows]
    result = metrics(records); save(output/'judgments.json',records); save(output/'summary.json',result)
    (output/'failures.json').unlink(missing_ok=True)
    return result


def choose_threshold(results):
    if any(v['localization_given_presence'] is None for v in results.values()):
        raise ValueError('Calibration needs at least one presence-positive concept')
    best = max(v['localization_given_presence'] for v in results.values())
    return min(float(k) for k,v in results.items() if v['localization_given_presence']==best)


def judge(args, client=None):
    folder, metadata, rows = load_export(args.export)
    if args.workers < 1: raise ValueError('workers must be positive')
    presence, localization = prompts(metadata['identity']['dataset'])
    output = args.output; output.mkdir(parents=True, exist_ok=True)
    with (output/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        protocol = dict(mode=args.command, export_sha256=file_digest(folder/'export.json'),
            identity=metadata['identity'], image_ids=metadata['image_ids'], model=args.model,
            expected_model=args.expected_model, base_url=args.base_url, workers=args.workers,
            presence_prompt=presence, localization_prompt=localization, response_format=SCHEMA,
            temperature=0, reasoning_effort='low', max_completion_tokens=2048, source_sha256=file_digest(__file__))
        if args.command == 'calibrate':
            thresholds = sorted(set(args.thresholds))
            if not thresholds or any(not 0<t<1 for t in thresholds): raise ValueError('Invalid threshold grid')
            protocol['thresholds'] = thresholds
            if metadata['split'] != 'validation': raise ValueError('Calibration requires validation, never test')
        else:
            frozen = read(args.calibration)
            calibration_protocol = read(args.calibration.parent/'protocol.json')
            done = read(args.calibration.parent/'complete.json')
            if not done['complete'] or done['frozen_threshold_sha256'] != file_digest(args.calibration):
                raise ValueError('Calibration is incomplete or threshold changed')
            if frozen['protocol_sha256'] != file_digest(args.calibration.parent/'protocol.json'):
                raise ValueError('Calibration protocol changed')
            if frozen['results_sha256'] != file_digest(args.calibration.parent/'calibration_results.json'):
                raise ValueError('Calibration results changed')
            if frozen['threshold'] != choose_threshold(read(args.calibration.parent/'calibration_results.json')):
                raise ValueError('Frozen threshold is not the calibration optimum')
            for key in ('identity','model','expected_model','base_url','presence_prompt','localization_prompt','response_format'):
                if protocol[key] != calibration_protocol[key]: raise ValueError('Calibration mismatch: '+key)
            if set(metadata['image_ids']) & set(calibration_protocol['image_ids']): raise ValueError('Calibration/test image overlap')
            if metadata['split'] != 'test': raise ValueError('Evaluation requires test export')
            thresholds = [frozen['threshold']]; protocol['calibration_sha256'] = file_digest(args.calibration)
        pin = output/'protocol.json'
        if pin.exists():
            if read(pin) != protocol: raise ValueError('Changed protocol; use a fresh output directory')
        else:
            if any(p.name != 'queue.lock' for p in output.iterdir()): raise ValueError('Nonempty unpinned output')
            save(pin,protocol)
        if (output/'complete.json').exists():
            done = read(output/'complete.json')
            if done['protocol_sha256'] != file_digest(pin): raise ValueError('Completion protocol changed')
            if done['summary_sha256'] != file_digest(output/'summary.json'): raise ValueError('Summary changed')
            if args.command == 'evaluate':
                trial = output/'thresholds'/str(thresholds[0])/'judgments.json'
                if done['judgments_sha256'] != file_digest(trial): raise ValueError('Judgments changed')
            return read(output/'summary.json')
        client = client or Client(args.base_url,args.model,args.expected_model,args.api_key_env,args.attempts,args.timeout)
        results = {}
        for threshold in thresholds:
            results[str(threshold)] = assess(folder,rows,metadata['identity']['dataset'],threshold,
                output/'thresholds'/str(threshold),client,output/'responses',args.workers)
        if args.command == 'calibrate':
            threshold = choose_threshold(results); save(output/'calibration_results.json',results)
            save(output/'frozen_threshold.json',dict(threshold=threshold,protocol_sha256=file_digest(pin),
                results_sha256=file_digest(output/'calibration_results.json')))
            result = results[str(threshold)]
        else: result = results[str(thresholds[0])]
        save(output/'summary.json',result)
        done = dict(complete=True,protocol_sha256=file_digest(pin),summary_sha256=file_digest(output/'summary.json'))
        if args.command == 'calibrate': done['frozen_threshold_sha256'] = file_digest(output/'frozen_threshold.json')
        else: done['judgments_sha256'] = file_digest(output/'thresholds'/str(thresholds[0])/'judgments.json')
        save(output/'complete.json',done)
        return result


def summarize(runs):
    protocols = [read(p/'protocol.json') for p in runs]
    keys = ('identity','image_ids','model','expected_model','base_url','calibration_sha256','export_sha256',
            'presence_prompt','localization_prompt','response_format')
    reference = protocols[0]
    summaries, records = [], []
    for path, protocol in zip(runs, protocols):
        if protocol['mode'] != 'evaluate' or any(protocol.get(k)!=reference.get(k) for k in keys):
            raise ValueError('Repeat inputs or judge settings differ')
        done = read(path/'complete.json')
        if (not done['complete'] or done['summary_sha256'] != file_digest(path/'summary.json') or
            done['protocol_sha256'] != file_digest(path/'protocol.json')):
            raise ValueError('Incomplete or changed repeat')
        summaries.append(read(path/'summary.json'))
        trials = list((path/'thresholds').glob('*/judgments.json'))
        if len(trials)!=1: raise ValueError('Expected one frozen-threshold trial')
        if done['judgments_sha256'] != file_digest(trials[0]): raise ValueError('Repeat judgments changed')
        records.append({r['id']:r for r in read(trials[0])})
    ids = set(records[0])
    if any(set(r)!=ids for r in records): raise ValueError('Repeat pair rosters differ')
    common = [i for i in ids if all(r[i]['presence']=='yes' for r in records)]
    def agreement(selected,field):
        return float(np.mean([len({r[i][field] for r in records})==1 for i in selected])) if selected else None
    def moments(key):
        values = [s[key] for s in summaries]
        if any(v is None for v in values): return dict(mean=None,sd=None)
        return dict(mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)) if len(values)>1 else None)
    return dict(runs=len(runs), metrics={k:moments(k)
        for k in ('concept_precision','localization_given_presence','grounded_concept_precision')},
        presence_unanimous_agreement=agreement(ids,'presence'), common_present_pairs=len(common),
        localization_unanimous_agreement_common_present=agreement(common,'localization'),
        note='Judge repeats, not independent training runs; conditional localization subsets can differ.')


def main():
    parser = argparse.ArgumentParser(description=__doc__); sub = parser.add_subparsers(dest='command',required=True)
    export_parser = sub.add_parser('export',help='Export native maps and frozen concepts from a release checkpoint')
    export_parser.add_argument('--config',type=Path,required=True); export_parser.add_argument('--head-dir',type=Path,required=True)
    export_parser.add_argument('--split',choices=['validation','test'],required=True)
    export_parser.add_argument('--count',type=int,required=True); export_parser.add_argument('--seed',type=int,default=0)
    export_parser.add_argument('--nec',type=int,default=5); export_parser.add_argument('--top-k',type=int,default=5)
    export_parser.add_argument('--output',type=Path,required=True)
    for command in ('calibrate','evaluate'):
        p = sub.add_parser(command); p.add_argument('--export',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
        p.add_argument('--base-url',required=True); p.add_argument('--model',required=True); p.add_argument('--expected-model',required=True)
        p.add_argument('--api-key-env',default='VLM_API_KEY'); p.add_argument('--workers',type=int,default=8)
        p.add_argument('--attempts',type=int,default=8); p.add_argument('--timeout',type=float,default=180)
        if command=='calibrate': p.add_argument('--thresholds',type=float,nargs='+',default=GRID)
        else: p.add_argument('--calibration',type=Path,required=True)
    p = sub.add_parser('summarize'); p.add_argument('--runs',nargs='+',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    if args.command=='export':
        from .vlm_export import export
        result = export(args)
    elif args.command=='summarize': result=summarize(args.runs); save(args.output,result)
    else: result=judge(args)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
