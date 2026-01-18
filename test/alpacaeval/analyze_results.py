"""
Analyze AlpacaEval 2.0 evaluation results.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re


def _safe_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _tokenize(value: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", value.lower()) if part]


def _matches_model(query: str, candidate: str) -> bool:
    query_tokens = _tokenize(query)
    if not query_tokens:
        return False
    candidate_lower = candidate.lower()
    return all(token in candidate_lower for token in query_tokens)


def _resolve_name_key(fieldnames: list[str]) -> str | None:
    if not fieldnames:
        return None
    for key in ("name", "model", "generator", "model_name"):
        if key in fieldnames:
            return key
    if "" in fieldnames:
        return ""
    return fieldnames[0]


def _to_fraction(rate: float | None) -> float | None:
    if rate is None:
        return None
    # AlpacaEval leaderboards typically store percentages (0-100), not fractions (0-1).
    if rate > 1.0:
        return rate / 100.0
    return rate


def _default_leaderboard_path(results_dir: str) -> str:
    history = os.path.join(results_dir, "leaderboard_history.csv")
    if os.path.exists(history):
        return history
    direct = os.path.join(results_dir, "leaderboard.csv")
    if os.path.exists(direct):
        return direct
    return direct


def _find_latest_file(results_dir: str, filename: str) -> str | None:
    latest_path: str | None = None
    latest_mtime: float = -1.0
    for root, _, files in os.walk(results_dir):
        if filename not in files:
            continue
        path = os.path.join(root, filename)
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = path
    return latest_path


def _print_leaderboard(leaderboard_path: str, model_name: str) -> None:
    if not os.path.exists(leaderboard_path):
        print(f"Leaderboard not found at {leaderboard_path}")
        return

    with open(leaderboard_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print("Leaderboard is empty.")
        return

    name_key = _resolve_name_key(fieldnames)
    model_row = None
    for row in rows:
        label = (row.get(name_key) if name_key is not None else None) or ""
        if _matches_model(model_name, label):
            model_row = row
            break

    print("\n" + "=" * 60)
    print("ALPACAEVAL 2.0 EVALUATION RESULTS")
    print("=" * 60)

    if model_row:
        label = (model_row.get(name_key) if name_key is not None else None) or "unknown"
        print(f"\nModel: {label}")
        win_rate = _to_fraction(_safe_float(model_row.get("win_rate")))
        if win_rate is not None:
            print(f"Win Rate: {win_rate:.2%}")

        lc_win = _to_fraction(_safe_float(model_row.get("length_controlled_winrate")))
        if lc_win is not None:
            print(f"Length-Controlled Win Rate: {lc_win:.2%}")

        avg_length = _safe_float(model_row.get("avg_length"))
        if avg_length is not None:
            print(f"Average Output Length: {avg_length:.0f} characters")
    else:
        print("\nModel not found in leaderboard. Showing full table.")

    print("\nTop 10 Models (by file order):")
    for row in rows[:10]:
        label = (row.get(name_key) if name_key is not None else None) or "unknown"
        win_rate = _to_fraction(_safe_float(row.get("win_rate")))
        win_rate_str = f"{win_rate:.2%}" if win_rate is not None else "n/a"
        print(f"- {label}: {win_rate_str}")


def _print_annotations(annotations_path: str) -> None:
    if not os.path.exists(annotations_path):
        print(f"\nAnnotations not found at {annotations_path}")
        return

    with open(annotations_path, "r", encoding="utf-8") as f:
        annotations = json.load(f)

    if not annotations:
        print("\nAnnotations file is empty.")
        return

    preferences = [item.get("preference", 0) for item in annotations]
    wins = preferences.count(2)
    losses = preferences.count(1)
    draws = preferences.count(0)
    total = len(preferences)

    print(f"\nTotal Evaluations: {total}")
    print(f"Wins: {wins} ({wins / total * 100:.1f}%)")
    print(f"Losses: {losses} ({losses / total * 100:.1f}%)")
    print(f"Draws: {draws} ({draws / total * 100:.1f}%)")
    print("\n" + "=" * 60 + "\n")


def analyze_results(results_dir: str, model_name: str) -> None:
    leaderboard_path = _default_leaderboard_path(results_dir)
    if not os.path.exists(leaderboard_path):
        maybe = _find_latest_file(results_dir, "leaderboard.csv")
        if maybe:
            leaderboard_path = maybe

    annotations_path = os.path.join(results_dir, "annotations.json")
    if not os.path.exists(annotations_path):
        sibling = os.path.join(os.path.dirname(leaderboard_path), "annotations.json")
        if os.path.exists(sibling):
            annotations_path = sibling
        else:
            maybe = _find_latest_file(results_dir, "annotations.json")
            if maybe:
                annotations_path = maybe
    print(f"Using leaderboard: {leaderboard_path}")
    print(f"Using annotations: {annotations_path}")
    _print_leaderboard(leaderboard_path, model_name)
    _print_annotations(annotations_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze AlpacaEval results")
    parser.add_argument(
        "--results_dir",
        type=str,
        default="test/alpacaeval/results",
        help="Directory containing evaluation results",
    )
    parser.add_argument(
        "--leaderboard_path",
        type=str,
        default=None,
        help="Optional path to leaderboard CSV (defaults to leaderboard_history.csv if present).",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="hh-llama32-1b-sft",
        help="Model name to locate in leaderboard",
    )
    args = parser.parse_args()
    if args.leaderboard_path:
        _print_leaderboard(args.leaderboard_path, args.model_name)
        _print_annotations(os.path.join(args.results_dir, "annotations.json"))
    else:
        analyze_results(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()
