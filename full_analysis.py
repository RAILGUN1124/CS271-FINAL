import os
import json
import glob
from collections import Counter, defaultdict
from tqdm import tqdm

def analyze():
    data_dir = "data"
    train_files = sorted(glob.glob(os.path.join(data_dir, "**/*train*.jsonl"), recursive=True))
    test_files = sorted(glob.glob(os.path.join(data_dir, "**/*test*.jsonl"), recursive=True))

    results = {}

    for split, files in [("train", train_files), ("test", test_files)]:
        stats = {
            "total_rows": 0,
            "labels": Counter(),
            "key_counts": Counter(),
            "null_counts": Counter(),
            "list_lengths": defaultdict(list),
            "dict_key_counts": defaultdict(list),
            "schemas": set()
        }
        
        for fpath in tqdm(files, desc=f"Processing {split} files", unit="file"):
            with open(fpath, 'r') as f:
                for line in f:
                    stats["total_rows"] += 1
                    data = json.loads(line)
                    
                    # Schema
                    keys = frozenset(data.keys())
                    stats["schemas"].add(keys)
                    
                    # Key presence and Label
                    for k, v in data.items():
                        stats["key_counts"][k] += 1
                        if k == 'label':
                            stats["labels"][v] += 1
                        
                        if v is None:
                            stats["null_counts"][k] += 1
                        elif isinstance(v, list):
                            stats["list_lengths"][k].append(len(v))
                        elif isinstance(v, dict):
                            stats["dict_key_counts"][k].append(len(v))
        
        results[split] = stats

    for split, stats in results.items():
        print(f"--- {split.upper()} ---")
        rows = stats["total_rows"]
        print(f"Total Rows: {rows}")
        print(f"Labels: {dict(stats['labels'])}")
        print(f"Schema consistency: {'Consistent' if len(stats['schemas']) == 1 else f'Inconsistent ({len(stats['schemas'])} variants)'}")
        
        print("Key Presence Rates:")
        for k, count in stats["key_counts"].items():
            print(f"  {k}: {count/rows:.2%}")
        
        print("Null Rates:")
        for k, count in stats["null_counts"].items():
            print(f"  {k}: {count/rows:.2%}")
            
        print("List Avg Lengths:")
        for k, lengths in stats["list_lengths"].items():
            avg = sum(lengths)/len(lengths) if lengths else 0
            print(f"  {k}: {avg:.2f}")
            
        print("Dict Avg Key Counts:")
        for k, counts in stats["dict_key_counts"].items():
            avg = sum(counts)/len(counts) if counts else 0
            print(f"  {k}: {avg:.2f}")
        print()

if __name__ == "__main__":
    analyze()
