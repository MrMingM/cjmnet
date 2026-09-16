"""Print a compact comparison from multiple GSPR diagnostic JSON files."""

import argparse
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+")
    args = parser.parse_args()
    header = ("condition", "points", "R_mean", "R_q05", "R_q50", "R_q95",
              "U_mean", "R<0.5", "corr_range", "corr_intensity", "corr_density")
    print("\t".join(header))
    for path in args.reports:
        with open(path, "r", encoding="utf-8") as stream:
            report = json.load(stream)
        condition = report.get("condition", path)
        reliability = report["reliability"]
        uncertainty = report["uncertainty"]
        correlations = report["pearson_reliability_correlation"]
        values = (
            condition, reliability["count"], reliability["mean"],
            reliability["q05"], reliability["median"], reliability["q95"],
            uncertainty["mean"], report["fraction_below"]["0.5"],
            correlations["range"], correlations["intensity"],
            correlations["density"])
        print("\t".join(str(value) for value in values))


if __name__ == "__main__":
    main()
