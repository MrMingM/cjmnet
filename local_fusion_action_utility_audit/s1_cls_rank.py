"""S1 classification source linear pairwise probe."""
from .common import parser
from .linear_ranker import evaluate, fit


def main():
    cli = parser(__doc__)
    cli.add_argument('--mode', choices=('fit', 'evaluate'), required=True)
    args = cli.parse_args()
    (fit if args.mode == 'fit' else evaluate)(args.run, 'cls')


if __name__ == '__main__':
    main()
