"""Entry script to run modularized DenseFusion preprocessing / training / eval."""
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
    main('complete')
