#!/usr/bin/env python3
import os
import sys

# Allow the target directory to be passed as a command-line argument.
if len(sys.argv) > 1:
    TARGET_DIR = sys.argv[1]
else:
    TARGET_DIR = "vllm-benchmarks"

# The index will list benchmark directories under TARGET_DIR.
INDEX_FILE = os.path.join(TARGET_DIR, "index.md")
TITLE = "# Inference Framework Benchmarks\n\n"

# Exclude any directories you don't want to list.
EXCLUDE_DIRS = {".github", ".git"}

def get_benchmark_directories():
    """
    Returns a list of directory names inside the TARGET_DIR
    that are assumed to be benchmark directories.
    """
    try:
        entries = os.listdir(TARGET_DIR)
    except FileNotFoundError:
        print(f"Error: The target directory '{TARGET_DIR}' does not exist!")
        return []

    directories = [
        entry for entry in entries
        if os.path.isdir(os.path.join(TARGET_DIR, entry)) and entry not in EXCLUDE_DIRS
    ]
    return directories

def generate_index_content(directories):
    """
    Generate index content with a title and a list of links for each benchmark directory.
    """
    directories.sort(reverse=True)
    content = TITLE
    for directory in directories:
        content += f"- [{directory}](./{directory})\n"
    return content

def main():
    benchmark_dirs = get_benchmark_directories()
    if not benchmark_dirs:
        print("No benchmark directories found. Exiting.")
        return

    index_content = generate_index_content(benchmark_dirs)
    with open(INDEX_FILE, "w") as f:
        f.write(index_content)

    print(f"Updated '{INDEX_FILE}' with {len(benchmark_dirs)} benchmark entries.")

if __name__ == "__main__":
    main()
