"""Entry script to run modularized DenseFusion preprocessing / training / eval."""
import argparse
from densefusion.config import config
from densefusion.train import train_densefusion
from densefusion.eval import run_complete_evaluation


def main(mode='complete'):
    config.print_config()
    if mode == 'train' or mode == 'complete':
        train_densefusion()
    if mode == 'eval' or mode == 'complete':
        run_complete_evaluation()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='DenseFusion 6D Pose Estimation Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python run_pipeline.py                    # train + eval (default)
  python run_pipeline.py --train            # train only
  python run_pipeline.py --eval             # eval only
        '''
    )
    parser.add_argument(
        '--train',
        action='store_true',
        help='Run training only'
    )
    parser.add_argument(
        '--eval',
        action='store_true',
        help='Run evaluation only'
    )
    
    args = parser.parse_args()
    
    # Determine mode based on arguments
    if args.train and args.eval:
        mode = 'complete'
    elif args.train:
        mode = 'train'
    elif args.eval:
        mode = 'eval'
    else:
        mode = 'complete'  # default: both train and eval
    
    main(mode)

