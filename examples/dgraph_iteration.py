import argparse

from opendg.graph import DGraph

parser = argparse.ArgumentParser(
    description='Iterate Dynamic Graph Dataset',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    '--dataset_file',
    type=str,
    default='data/foo.csv',
    help='Path to dyamic graph dataset.',
)


def main() -> None:
    args = parser.parse_args()

    DG = DGraph.from_csv(args.dataset_file)
    print(DG)


if __name__ == '__main__':
    main()
