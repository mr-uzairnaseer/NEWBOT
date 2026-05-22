import json

log_path = r"C:\Users\dev\.gemini\antigravity\brain\8bd20619-b1e5-4df4-a213-b7a0146c3ca7\.system_generated\logs\overview.txt"
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        try:
            data = json.loads(line)
            if data.get("step_index") == 48:
                print("--- STEP 48 USER INPUT ---")
                print(data.get("content"))
                print("--------------------------")
        except Exception as e:
            print(f"Error parsing line: {e}")
