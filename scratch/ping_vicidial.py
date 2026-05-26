import sys
import paramiko

try:
    sys.stdout.reconfigure(encoding='utf-8')
except AttributeError:
    pass

HOST = "37.27.133.33"
PORT = 22
USER = "root"
PASS = "UFfzwghHyKkSyK"

def run_cmd(ssh, cmd):
    print(f"Executing: {cmd}")
    stdin, stdout, stderr = ssh.exec_command(cmd)
    out = stdout.read().decode('utf-8', errors='replace').strip()
    err = stderr.read().decode('utf-8', errors='replace').strip()
    if out:
        print(f"[STDOUT]\n{out}")
    if err:
        print(f"[STDERR]\n{err}")
    print("-" * 40)

def main():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username=USER, password=PASS)
    
    print("=== Pinging ViciDial Peer 1 (88.99.209.24) ===")
    run_cmd(ssh, "ping -c 3 88.99.209.24")
    
    print("=== Pinging ViciDial Peer 2 (138.201.58.110) ===")
    run_cmd(ssh, "ping -c 3 138.201.58.110")
    
    print("=== Checking if UDP Port 5060 on ViciDial is responsive ===")
    run_cmd(ssh, "nc -zuv -w 3 88.99.209.24 5060")
    run_cmd(ssh, "nc -zuv -w 3 138.201.58.110 5060")
    
    ssh.close()

if __name__ == "__main__":
    main()
