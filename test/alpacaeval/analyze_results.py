"""
Analyze AlpacaEval evaluation results.
"""

import argparse
import csv
import json
import os


DEFAULT_RESULTS_DIR = "test/alpacaeval/results"


def read_leaderboard(path):
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def analyze_results(results_dir, model_name):
    leaderboard_path = os.path.join(results_dir, "leaderboard.csv")
    if not os.path.exists(leaderboard_path):
        print(f"Leaderboard not found: {leaderboard_path}")
        return

    leaderboard = read_leaderboard(leaderboard_path)
    model_rows = [
        row
        for row in leaderboard
        if model_name in row.get("name", "") or model_name in row.get("model", "")
    ]

    print("\n" + "=" * 60)
    print("ALPACAEVAL RESULTS")
    print("=" * 60)

    if model_rows:
        row = model_rows[0]
        name = row.get("name", model_name)
        win_rate = row.get("win_rate")
        lc_winrate = row.get("length_controlled_winrate")
        avg_length = row.get("avg_length")

        print(f"\nModel: {name}")
        if win_rate is not None:
            print(f"Win Rate: {float(win_rate):.2%}")
        if lc_winrate is not None:
            print(f"Length-Controlled Win Rate: {float(lc_winrate):.2%}")
        if avg_length is not None:
            print(f"Average Output Length: {float(avg_length):.0f} characters")
    else:
        print("\nModel not found in leaderboard. Showing top 10 entries:")
        for row in leaderboard[:10]:
            print(f"- {row.get('name', 'unknown')}: {row.get('win_rate', 'n/a')}")

    annotations_path = os.path.join(results_dir, "annotations.json")
    if os.path.exists(annotations_path):
        with open(annotations_path, "r", encoding="utf-8") as f:
            annotations = json.load(f)

        preferences = [a.get("preference", 0) for a in annotations]
        wins = preferences.count(2)
        losses = preferences.count(1)
        draws = preferences.count(0)

        print(f"\nTotal Evaluations: {len(annotations)}")
        print(f"Wins: {wins} ({wins/len(annotations)*100:.1f}%)")
        print(f"Losses: {losses} ({losses/len(annotations)*100:.1f}%)")
        print(f"Draws: {draws} ({draws/len(annotations)*100:.1f}%)")
    else:
        print(f"\nAnnotations file not found: {annotations_path}")

    print("\n" + "=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze AlpacaEval results")
    parser.add_argument("--results_dir", type=str, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--model_name", type=str, default="hh-llama32-1b-sft")
    args = parser.parse_args()

    analyze_results(args.results_dir, args.model_name)


if __name__ == "__main__":
    main()
