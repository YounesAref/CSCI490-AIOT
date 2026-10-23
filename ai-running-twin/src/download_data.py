"""Stream complete running workouts from the official FitRec filtered export."""
import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_URL = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/fitrec/endomondoHR_proper.json"


def parse_record(line):
    # Official .json is actually one Python dictionary literal per line.
    # literal_eval accepts literals only; never execute dataset content with eval.
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    try:
        value = json.loads(line)
    except json.JSONDecodeError:
        value = ast.literal_eval(line)
    if not isinstance(value, dict):
        raise ValueError("Expected one workout dictionary per line")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--input", type=Path, help="Read a local official .json or .gz export")
    parser.add_argument("--workouts", type=int, default=200)
    parser.add_argument("--max-records", type=int, default=20000)
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/running_workouts.jsonl")
    args = parser.parse_args()
    if args.workouts < 2 or args.max_records < args.workouts:
        parser.error("Need at least two workouts and max-records >= workouts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_suffix(".partial")
    sports, seen = Counter(), set()
    scanned = selected = duplicates = 0
    try:
        if args.input:
            opener = gzip.open if args.input.suffix == ".gz" else open
            source = opener(args.input, "rb")
        else:
            source = urllib.request.urlopen(args.url, timeout=60)
        with source, partial.open("w") as dest:
            for line in source:
                if not line.strip():
                    continue
                record = parse_record(line)
                scanned += 1
                sport = str(record.get("sport", "missing")).strip().lower()
                sports[sport] += 1
                if sport == "run":
                    key = str(record.get("id", ""))
                    if key and key not in seen:
                        seen.add(key)
                        dest.write(json.dumps(record, allow_nan=False) + "\n")
                        selected += 1
                        if selected % 50 == 0:
                            print(f"Selected {selected} runs after scanning {scanned} workouts", flush=True)
                    else:
                        duplicates += 1
                if selected >= args.workouts or scanned >= args.max_records:
                    break
        if selected < 2:
            raise ValueError(f"Only {selected} running workouts found; need at least two")
        partial.replace(args.output)
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    manifest = {
        "source_url": args.url, "local_input": str(args.input) if args.input else None,
        "selection": "First unique complete workouts with sport exactly 'run', in source order; not a random sample",
        "requested_running_workouts": args.workouts, "selected_running_workouts": selected,
        "scanned_workouts": scanned, "scanned_sport_counts": dict(sports),
        "duplicate_or_missing_running_ids_skipped": duplicates,
        "max_records": args.max_records,
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "bytes_saved": args.output.stat().st_size,
    }
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
