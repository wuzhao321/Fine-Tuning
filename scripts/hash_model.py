"""Hash a local model snapshot and check indexed shards; no network access."""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from llm_ft_repro.evidence import sha256_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    index = json.loads((args.snapshot / "model.safetensors.index.json").read_text())
    required = set(index["weight_map"].values()) | {"config.json", "tokenizer_config.json",
                                                  "model.safetensors.index.json"}
    for name in required:
        if not (args.snapshot / name).is_file():
            raise FileNotFoundError(name)
    manifest = {"snapshot": str(args.snapshot.resolve()), "files": {
        str(p.relative_to(args.snapshot)): {"bytes": p.stat().st_size, "sha256": sha256_file(p)}
        for p in sorted(args.snapshot.rglob("*")) if p.is_file() and ".cache" not in p.parts}}
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
