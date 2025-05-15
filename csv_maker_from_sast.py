import os
import json
import pandas as pd

# Configuration
CPP_ISSUES_FILE = "bigvul_cpp_issues.json"
CSV_FILE = "bigvul_updated_with_cpp_target.csv"


def load_json(filename):
    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def load_csv(filename):
    return pd.read_csv(filename)


def extract_indexes_from_json(issues_data):
    indexes = set()
    for issue in issues_data:
        component = issue.get("component", "")
        file_part = component.split(":")[-1]
        file_part = file_part.replace(".c", "")  # Remove .c extension if present
        index = file_part.split("_")[0]
        if index.isdigit():
            indexes.add(int(index))
    return indexes



def mark_target_cppcheck(csv_data, issue_indexes):
    csv_data["target_cppcheck"] = 0
    for idx in issue_indexes:
        if 0 <= idx < len(csv_data):
            csv_data.at[idx, "target_cppcheck"] = 1
    print(f"✅ Set 'target_cppcheck' = 1 for {csv_data['target_cppcheck'].sum()} rows.")


if __name__ == "__main__":
    print("🔍 Loading CSV file...")
    df = pd.read_csv(CSV_FILE)

    print("🧹 Dropping old 'target_cppcheck' column if it exists...")
    if "target_cppcheck" and "matches_func" in df.columns:
        df = df.drop(columns=["target_cppcheck"])
        df = df.drop(columns=["matches_func"])
    df.to_csv(CSV_FILE, index=False)

    print("🔁 Reloading cleaned CSV...")
    csv_data = load_csv(CSV_FILE)

    print("📦 Loading JSON issues...")
    cpp_issues = load_json(CPP_ISSUES_FILE)
    issue_indexes = extract_indexes_from_json(cpp_issues)
    print(f"✅ Found {len(issue_indexes)} unique indexes in JSON.")

    print("🎯 Marking 'target_cppcheck' in CSV...")
    mark_target_cppcheck(csv_data, issue_indexes)

    print("Marking 'target_cppcheck' in CSV as 0 for the rest of the rows...")
    csv_data["target_cppcheck"] = csv_data["target_cppcheck"].fillna(0)

    #are there any rows with NaN values?
    print("Checking for NaN values...")
    print(csv_data.isnull().sum())

    print("💾 Saving updated CSV...")
    csv_data.to_csv(CSV_FILE, index=False)
    print("✅ Done.")
