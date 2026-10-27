import argparse
import json

from .config import load


def main():
    parser = argparse.ArgumentParser(description='Shared SG-CBM release training')
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest')
    manifest.add_argument('--layout', choices=['cub','imagefolder','places365','manifest'], required=True)
    manifest.add_argument('--root', required=True)
    manifest.add_argument('--output', required=True)
    manifest.add_argument('--filelist')
    manifest.add_argument('--annotation-map')
    manifest.add_argument('--partition',choices=['train','test'],default='train')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--dataset', choices=['cub','imagenet','places365','partimagenetpp'], required=True)
    prepare.add_argument('--concepts')
    prepare.add_argument('--split')
    prepare.add_argument('--test-source', required=True)
    prepare.add_argument('--fraction', type=float)
    prepare.add_argument('--seed', type=int, default=6885)
    for name in ['check','train','export','calibrate','evaluate']:
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        if name == 'train':
            command.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.command == 'manifest':
        from .manifest import create
        create(args.layout,args.root,args.output,args.filelist,args.annotation_map,args.partition)
        return
    if args.command == 'prepare':
        from .data import prepare
        result = prepare(args.source,args.output,args.dataset,args.concepts,args.split,args.fraction,args.seed,args.test_source)
    else:
        cfg = load(args.config)
        if args.command == 'check':
            from pathlib import Path
            from .data import verify
            result = dict(config=cfg.asdict(), prepared=verify(cfg.data.prepared))
            if not result['prepared'].get('test_disjoint_verified'):
                raise ValueError('Held-out test isolation has not been checked')
            if cfg.dataset != result['prepared']['dataset']:
                raise ValueError('Prepared dataset differs from config')
            if cfg.model.checkpoint and not Path(cfg.model.checkpoint).is_file():
                raise FileNotFoundError(cfg.model.checkpoint)
            if cfg.training.stage == 'concept':
                prepared = result['prepared']
                if prepared['annotated_images'] != prepared['images'] or not (Path(cfg.data.prepared)/'concepts.txt').read_text().strip():
                    raise ValueError('Concept training requires a vocabulary and annotations for every image')
        elif args.command == 'train':
            from .engine import fit
            result = fit(cfg,args.resume)
        elif args.command in ('calibrate','evaluate'):
            from .metrics import assess
            result = assess(cfg,test=args.command=='evaluate')
        else:
            from .export import export
            result = export(cfg)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
