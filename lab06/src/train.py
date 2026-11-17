import argparse
from pathlib import Path

from common import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path)
    parser.add_argument('--task', choices=['train', 'workers', 'profile'], default='train')
    parser.add_argument('--name', default='baseline')
    parser.add_argument('--study', default='single')
    parser.add_argument('--stop-after', type=int)
    parser.add_argument('--resume', type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.task == 'workers':
        from studies import worker_study
        selected = worker_study(config)
        print(f'Selected workers: {selected}', flush=True)
    elif args.task == 'profile':
        from checks import profile_epoch
        profile_epoch(config)
    else:
        import json
        from common import RESULTS
        from engine import train
        result, _ = train(config, args.name, args.study, args.stop_after, args.resume)
        path = RESULTS / 'runs' / f'{args.name}.json'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
