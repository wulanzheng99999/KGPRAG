"""
Build a de-duplicated corpus from hotpot_dev_distractor_v1.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_corpus(input_path: Path) -> list[dict[str, str]]:
    with input_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    unique_docs: dict[str, str] = {}
    for sample in data:
        for title, sentences in sample.get("context", []):
            if title in unique_docs:
                continue
            unique_docs[title] = " ".join(sentences)

    return [{"title": title, "text": text} for title, text in unique_docs.items()]


def main() -> None:
    # Determine the project root directory relative to this script
    # Script is in <root>/util/, so parent.parent is <root>
    project_root = Path(__file__).resolve().parent.parent
    default_input = project_root / "data" / "hotpot_dev_distractor_v1.json"
    default_output = project_root / "data" / "hotpot_dev_fullwiki_v2.json"

    parser = argparse.ArgumentParser(
        description="Build corpus from hotpot_dev_distractor_v1.json (dedupe by title)."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=default_input,
        help="Path to hotpot_dev_distractor_v1.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output corpus JSON file.",
    )
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    corpus = build_corpus(input_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False)

    print(f"Corpus docs: {len(corpus)}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
