import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

from tqdm import tqdm


def find_jsonl_files(root_dir: Path) -> List[Path]:
    """Recursively find every .jsonl file under the input directory."""
    return sorted(
        path for path in root_dir.rglob("*") if path.is_file() and path.suffix.lower() == ".jsonl"
    )


def dedupe_jsonl_file(file_path: Path) -> Tuple[str, int, int, bool]:
    """Remove duplicate rows within one JSONL file and rewrite it if needed."""
    seen = set()
    unique_lines: List[str] = []
    total_rows = 0

    with file_path.open("r", encoding="utf-8") as handle:
        for line_num, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            total_rows += 1
            line_without_newline = raw_line.rstrip("\r\n")

            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_num} in {file_path}") from exc

            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected JSON object per line in {file_path}, line {line_num}, got {type(row).__name__}"
                )

            md5_value = row.get("md5")
            if md5_value is None:
                unique_lines.append(line_without_newline)
                continue

            row_key = md5_value
            if row_key in seen:
                continue

            seen.add(row_key)
            unique_lines.append(line_without_newline)

    duplicate_count = total_rows - len(unique_lines)
    changed = duplicate_count > 0

    if changed:
        with file_path.open("w", encoding="utf-8") as handle:
            for line in unique_lines:
                handle.write(line)
                handle.write("\n")

    return str(file_path), total_rows, duplicate_count, changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Recursively scan JSONL files, remove duplicate rows within each file, and rewrite files in place."
        )
    )
    parser.add_argument("--input-dir", default="data", help="Root directory to scan")
    parser.add_argument("--max-workers", type=int, default=5, help="Maximum number of worker processes")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    files = find_jsonl_files(input_dir)
    if not files:
        print("No .jsonl files found.")
        return

    max_workers = max(1, min(args.max_workers, 8, os.cpu_count() or 1))

    processed = 0
    rewritten = 0
    duplicate_rows_removed = 0

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(dedupe_jsonl_file, file_path) for file_path in files]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Deduplicating", unit="file"):
            file_path, total_rows, duplicate_count, changed = future.result()
            processed += 1
            duplicate_rows_removed += duplicate_count
            if changed:
                rewritten += 1
                print(f"Rewrote {file_path}: rows={total_rows} duplicates_removed={duplicate_count}")

    print("Done")
    print(f"Files processed: {processed}")
    print(f"Files rewritten: {rewritten}")
    print(f"Duplicate rows removed: {duplicate_rows_removed}")


if __name__ == "__main__":
    main()