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
    
    print("=== Checking UFW Status ===")
    run_cmd(ssh, "ufw status")
    
    print("=== Allowing Port 5060 UDP in UFW ===")
    run_cmd(ssh, "ufw allow 5060/udp")
    run_cmd(ssh, "ufw allow 5060/tcp")
    run_cmd(ssh, "ufw allow 10000:20000/udp") # Standard RTP audio range
    
    print("=== Reloading UFW ===")
    run_cmd(ssh, "ufw reload")
    run_cmd(ssh, "ufw status verbose")
    
    ssh.close()

if __name__ == "__main__":
    main()
