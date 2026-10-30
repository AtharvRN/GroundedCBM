import argparse
import json

from .config import load


def main():
    parser = argparse.ArgumentParser(description='Shared spatial-CBM and baseline training')
    commands = parser.add_subparsers(dest='command', required=True)
    manifest = commands.add_parser('manifest')
    manifest.add_argument('--layout', choices=['cub','imagefolder','places365','manifest'], required=True)
    manifest.add_argument('--root', required=True)
    manifest.add_argument('--output', required=True)
    manifest.add_argument('--filelist')
    manifest.add_argument('--annotation-map')
    manifest.add_argument('--partition',choices=['train','test'],default='train')
    manifest.add_argument('--human-annotations')
    manifest.add_argument('--category-map')
    manifest.add_argument('--class-map',help='Saved training class-name-to-ID JSON for ImageFolder evaluation')
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--source', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--dataset', choices=['cub','imagenet','places365','partimagenetpp'], required=True)
    prepare.add_argument('--concepts')
    prepare.add_argument('--split')
    prepare.add_argument('--test-source', required=True)
    prepare.add_argument('--supervision',choices=['gdino','human'],default='gdino')
    prepare.add_argument('--fraction', type=float)
    prepare.add_argument('--seed', type=int, default=6885)
    for name in ['check','train','export','calibrate','evaluate','classify','teacher']:
        command = commands.add_parser(name)
        command.add_argument('--config', required=True)
        if name == 'train':
            command.add_argument('--resume', action='store_true')
        if name == 'classify':
            command.add_argument('--classifier-dir',required=True)
            command.add_argument('--nec-values',default='5,10,15,20,25,30')
    args = parser.parse_args()
    if args.command == 'manifest':
        from .manifest import create
        create(args.layout,args.root,args.output,args.filelist,args.annotation_map,args.partition,args.human_annotations,args.category_map,args.class_map)
        return
    if args.command == 'prepare':
        from .data import prepare
        result = prepare(args.source,args.output,args.dataset,args.concepts,args.split,args.fraction,args.seed,args.test_source,args.supervision)
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
            if result['prepared'].get('supervision','gdino') != cfg.data.supervision:
                raise ValueError('Prepared supervision differs from config')
            if cfg.model.checkpoint and not Path(cfg.model.checkpoint).is_file():
                raise FileNotFoundError(cfg.model.checkpoint)
            if cfg.training.stage == 'concept':
                prepared = result['prepared']
                if not (Path(cfg.data.prepared)/'concepts.txt').read_text().strip():
                    raise ValueError('Concept training requires a vocabulary')
                if cfg.model.method in {'sg_cbm','vlg_cbm'} and prepared['annotated_images'] != prepared['images']:
                    raise ValueError('Concept training requires a vocabulary and annotations for every image')
                if cfg.model.method in {'lf_cbm','salf_cbm'}:
                    from .teacher import verify_teacher
                    result['teacher'] = verify_teacher(cfg,prepared)
        elif args.command == 'teacher':
            from .teacher import prepare_teacher
            result = prepare_teacher(cfg)
        elif args.command == 'train':
            from .engine import fit
            result = fit(cfg,args.resume)
        elif args.command in ('calibrate','evaluate'):
            from .metrics import assess
            result = assess(cfg,test=args.command=='evaluate')
        elif args.command == 'classify':
            from .classifier import evaluate, parse_nec_values
            result = evaluate(cfg,args.classifier_dir,parse_nec_values(args.nec_values))
        else:
            from .export import export
            result = export(cfg)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
