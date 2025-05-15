import pandas as pd
import os
import argparse

def extract_functions(input_csv, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(input_csv)

    # Try to locate the function/code column
    possible_func_cols = ['func', 'function', 'code', 'func_before', 'func_after']
    func_col = None

    for col in df.columns:
        if col.lower() in possible_func_cols:
            func_col = col
            break

    if not func_col:
        raise ValueError(f"Could not find a function column in: {df.columns}")

    print(f"Using column '{func_col}' as source of function code.")

    for idx, row in df.iterrows():
        code = row[func_col]
        if not isinstance(code, str) or len(code.strip()) == 0:
            continue

        file_path = os.path.join(output_dir, f"func_{idx}.c")
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(code)

    print(f"Exported {len(df)} functions to '{output_dir}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert BigVul dataset functions to .c files.")
    parser.add_argument("--input", required=True, help="Path to bigvul.csv")
    parser.add_argument("--output", required=True, help="Output directory for .c files")
    args = parser.parse_args()

    extract_functions(args.input, args.output)
