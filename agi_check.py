#!/usr/bin/env python3
import sys
import os
import psycopg2

def main():
    # AGI sends logs and variables on stdin, and reads command results on stdin.
    # We must read stdin until empty to clear the initial AGI environment variables sent by Asterisk.
    agi_env = {}
    while True:
        line = sys.stdin.readline().strip()
        if not line:
            break
        if "=" in line:
            key, val = line.split(":", 1)
            agi_env[key.strip()] = val.strip()

    # The unique ID of the call is passed as the first argument to the AGI script.
    # If not present, try to extract it from the AGI environment.
    uniqueid = sys.argv[1] if len(sys.argv) > 1 else agi_env.get("agi_uniqueid")
    
    if not uniqueid:
        sys.stderr.write("Error: No uniqueid provided to AGI script.\n")
        # Default to DROP to be safe
        sys.stdout.write("SET VARIABLE LEAD_ACTION DROP\n")
        sys.stdout.flush()
        return

    # Load environment variables from /opt/voicebot/.env if it exists
    env_path = "/opt/voicebot/.env"
    if os.path.exists(env_path):
        with open(env_path) as f:
            for line in f:
                if "=" in line and not line.startswith("#"):
                    k, v = line.strip().split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

    # Connect to PostgreSQL using environment credentials
    db_user = os.getenv("DB_USER", "postgres")
    db_pass = os.getenv("DB_PASSWORD", "secret")
    db_host = os.getenv("DB_HOST", "127.0.0.1")
    db_port = os.getenv("DB_PORT", "5432")
    db_name = os.getenv("DB_NAME", "voicebot_db")

    disposition = "DROP" # Default fallback
    
    try:
        conn = psycopg2.connect(
            user=db_user,
            password=db_pass,
            host=db_host,
            port=db_port,
            database=db_name
        )
        cur = conn.cursor()
        cur.execute("SELECT disposition FROM call_logs WHERE uniqueid = %s;", (uniqueid,))
        row = cur.fetchone()
        if row:
            disposition = row[0]
        cur.close()
        conn.close()
    except Exception as e:
        sys.stderr.write(f"Database error in AGI: {e}\n")

    # Write the command back to Asterisk via stdout
    sys.stdout.write(f'SET VARIABLE LEAD_ACTION "{disposition}"\n')
    sys.stdout.flush()

if __name__ == "__main__":
    main()
