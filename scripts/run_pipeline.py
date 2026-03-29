from __future__ import annotations

import argparse
import json

from fig2csv.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="Path to chart image")
    parser.add_argument("--output-dir", default="outputs")
    args = parser.parse_args()

    result = run_pipeline(args.image, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
