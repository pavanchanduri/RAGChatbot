import json
import os

KB_FILE = "fix_history.json"

def save_fix(test_name, error, fix):
    record = {"test_name": test_name, "error": error, "fix": fix}
    if not os.path.exists(KB_FILE):
        with open(KB_FILE, "w") as f:
            json.dump([record], f, indent=2)
    else:
        with open(KB_FILE, "r") as f:
            data = json.load(f)
        data.append(record)
        with open(KB_FILE, "w") as f:
            json.dump(data, f, indent=2)

def get_past_fixes(error):
    if not os.path.exists(KB_FILE):
        return []
    with open(KB_FILE, "r") as f:
        data = json.load(f)
    return [rec["fix"] for rec in data if error in rec["error"]]