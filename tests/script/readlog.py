import argparse
import sys

def read_and_filter_logs(smid, log_file="debug.log"):
    """
    Read logs from a file and filter by the given smid.
    
    Args:
        smid: The ID to filter logs by
        log_file: Path to the log file (default: debug.log)
    """
    prefix = f"[{smid}]"
    
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if line.startswith(prefix):
                    print(line.rstrip())
                    if "branch" in line:
                        print()
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found", file=sys.stderr)
        sys.exit(1)

def search_logs_for_smids(smid, search, log_file="debug.log"):
    try:
        smid_limit = int(smid)
    except ValueError:
        print(f"Error: smid '{smid}' is not an integer", file=sys.stderr)
        sys.exit(1)

    found_smids = set()
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if search not in line or not line.startswith('['):
                    continue
                end = line.find(']')
                if end == -1:
                    continue
                smid_str = line[1:end]
                if not smid_str.isdigit():
                    continue
                smid_val = int(smid_str)
                if 0 <= smid_val < smid_limit:
                    found_smids.add(smid_val)
                    if len(found_smids) == smid_limit:
                        break
    except FileNotFoundError:
        print(f"Error: File '{log_file}' not found", file=sys.stderr)
        sys.exit(1)

    for missing_smid in range(smid_limit):
        if missing_smid not in found_smids:
            print(f"Error: '{search}' not found for smid {missing_smid}", file=sys.stderr)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter logs by smid from debug.log")
    parser.add_argument("smid", help="The smid to filter logs by")
    parser.add_argument("--log-file", "-l", default="debug.log", help="Path to the log file (default: debug.log)")
    parser.add_argument("--search", "-s", help="Search for a string in logs for each smid in range [0, smid)")

    
    args = parser.parse_args()
    if args.search:
        search_logs_for_smids(args.smid, args.search, args.log_file)
    else:
        read_and_filter_logs(args.smid, args.log_file)
