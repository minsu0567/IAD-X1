import argparse
import json
import os
import random

BENCHMARKS = ["VisA", "SDD", "MPDD", "DTD", "DS-MVTec", "DAGM"]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def resolve_reference_path(benchmark, query_rel_path, text_gt, data_root):
    parts = query_rel_path.split("/")
    if len(parts) < 3:
        return None
    good_dir_rel = "/".join(parts[:-2] + ["good"])
    good_dir_abs = os.path.join(data_root, good_dir_rel)
    if not os.path.isdir(good_dir_abs):
        return None

    files = [f for f in os.listdir(good_dir_abs) if f.lower().endswith(IMAGE_EXTS)]
    if not files:
        return None
    files.sort(reverse=True)

    chosen = files[0]
    if parts[-2] == "good" and chosen == parts[-1]:
        if len(files) < 2:
            return None
        chosen = files[1]
    return os.path.join(good_dir_abs, chosen)


def pick(pool, n, benchmark, data_root, chat_ad, rng):
    shuffled = list(pool)
    rng.shuffle(shuffled)

    picked = []
    for rel in shuffled:
        if len(picked) == n:
            break
        query_abs = os.path.join(data_root, rel)
        if not os.path.isfile(query_abs):
            continue
        ref_abs = resolve_reference_path(benchmark, rel, chat_ad[rel], data_root)
        if ref_abs is None:
            continue
        picked.append({
            "benchmark": benchmark,
            "query_path": query_abs,
            "ref_path": ref_abs,
            "is_normal": "good" in rel,
            "query_rel": rel,
        })
    return picked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str,
                    default="/content/drive/MyDrive/Uni-IAD_eval_dataset",
                    help="root holding <Benchmark>/test_<Benchmark>_format.json + images")
    ap.add_argument("--out", type=str,
                    default="/content/drive/MyDrive/throughput_manifest_180.json")
    ap.add_argument("--per_benchmark", type=int, default=30,
                    help="samples per benchmark; split evenly between normal and abnormal")
    ap.add_argument("--seed", type=int, default=6666)
    args = ap.parse_args()

    if args.per_benchmark % 2 != 0:
        raise SystemExit("--per_benchmark must be even (it is split 50/50 normal/abnormal).")
    half = args.per_benchmark // 2

    rng = random.Random(args.seed)
    manifest = []

    for benchmark in BENCHMARKS:
        json_path = os.path.join(args.data_root, benchmark, f"test_{benchmark}_format.json")
        if not os.path.isfile(json_path):
            raise SystemExit(f"Benchmark JSON not found: {json_path}")
        with open(json_path, "r", encoding="utf-8") as f:
            chat_ad = json.load(f)

        normal = [k for k in chat_ad if "good" in k]
        abnormal = [k for k in chat_ad if "good" not in k]

        got_n = pick(normal, half, benchmark, args.data_root, chat_ad, rng)
        got_a = pick(abnormal, half, benchmark, args.data_root, chat_ad, rng)

        if len(got_n) < half or len(got_a) < half:
            raise SystemExit(
                f"{benchmark}: only got {len(got_n)}/{half} normal and {len(got_a)}/{half} "
                f"abnormal samples with both query and reference present on disk "
                f"(pool: {len(normal)} normal / {len(abnormal)} abnormal). "
                f"Check --data_root images."
            )

        manifest.extend(got_n + got_a)
        print(f"{benchmark:10s} {len(got_n)} normal + {len(got_a)} abnormal")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1, ensure_ascii=False)

    print("-" * 60)
    print(f"{len(manifest)} samples -> {args.out}")
    print(f"  normal   : {sum(1 for m in manifest if m['is_normal'])}")
    print(f"  abnormal : {sum(1 for m in manifest if not m['is_normal'])}")
    print("All three throughput notebooks read this file; rebuild it and they all move together.")


if __name__ == "__main__":
    main()
