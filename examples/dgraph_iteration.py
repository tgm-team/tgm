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


def run_graph_iteration(dataset_file: str) -> None:
    DG = DGraph.from_csv(dataset_file)
    print(DG)


def main() -> None:
    args = parser.parse_args()
    run_graph_iteration(args.dataset_file)


if __name__ == '__main__':
    main()
