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
    
    print("=== Writing UUID-compatible extensions.conf to Server B ===")
    
    extensions_conf = """; ==============================================================================
; ASTERISK DIALPLAN - AI VOICE BOT AND LEAD TRANSFER ROUTING
; ==============================================================================

[general]
static=yes
writeprotect=no

[from-vicidial]
; Catch-all for dialed numbers and direct trunk calls
exten => s,1,Goto(_X.,1)

; ── Telephony Engineer Explicit Test Extensions ──
exten => 6666,1,NoOp(=== Telephony Engineer Dialed Test Extension 6666 ===)
same => n,Goto(_X.,1)

exten => 7777,1,NoOp(=== Telephony Engineer Dialed Test Extension 7777 ===)
same => n,Goto(_X.,1)

exten => 77777,1,NoOp(=== Telephony Engineer Dialed Test Extension 77777 ===)
same => n,Goto(_X.,1)

; Catch-all matching any dialed string of digits
exten => _X.,1,NoOp(=== INCOMING VOICEBOT CALL FOR EXTENSION ${EXTEN} ===)
same => n,Answer()
same => n,Playback(silence/1)  ; Let SIP audio channel establish perfectly

; Generate a valid RFC 4122 compliant UUID (128-bit)
; Required because Asterisk app_audiosocket strictly rejects non-RFC UUIDs (like UNIQUEID)
same => n,Set(CALL_UUID=${UUID()})

; Extract ViciDial Campaign and Lead IDs from custom SIP headers
; Supports BOTH chan_sip (SIP_HEADER) and chan_pjsip (PJSIP_HEADER)
same => n,Set(VICI_LEAD_ID=${SIP_HEADER(X-VICIdial-Lead-Id)})
same => n,Set(VICI_CAMP_NAME=${SIP_HEADER(X-VICIdial-Campaign-Id)})

same => n,GotoIf($["${VICI_LEAD_ID}" != ""]?has_headers)
same => n,Set(VICI_LEAD_ID=${PJSIP_HEADER(read,X-VICIdial-Lead-Id)})
same => n,Set(VICI_CAMP_NAME=${PJSIP_HEADER(read,X-VICIdial-Campaign-Id)})

same => n(has_headers),NoOp(Extracted ViciDial Lead ID: ${VICI_LEAD_ID} | Campaign: ${VICI_CAMP_NAME})

; Tunnel audio bidirectionally to Python AudioSocket Bridge running on port 9092
; We pass our newly generated compliant CALL_UUID
same => n,AudioSocket(${CALL_UUID},127.0.0.1:9092)
same => n,NoOp(AudioSocket channel session terminated.)

; Run the AGI status auditor to check if lead was qualified for transfer
; We pass the same CALL_UUID to match PostgreSQL records
same => n,AGI(/opt/voicebot/agi_check.py,${CALL_UUID})
same => n,NoOp(AI Call Audit disposition result: ${LEAD_ACTION})

; Route the call based on lead qualification outcome
same => n,GotoIf($["${LEAD_ACTION}" = "TRANSFER"]?transfer_lead:hangup_call)

; ── Route A: Transfer Qualified Lead ──
same => n(transfer_lead),NoOp(Lead QUALIFIED! Routing call back to ViciDial Closer queue...)
; Check if dial is over chan_sip or chan_pjsip
same => n,GotoIf($["${CHANNEL(channeltype)}" = "SIP"]?dial_sip:dial_pjsip)

same => n(dial_sip),Dial(SIP/transfer-closer@Testpeers,30,tT)
same => n,Hangup()

same => n(dial_pjsip),Dial(PJSIP/transfer-closer@vicidial-trunk,30,tT)
same => n,Hangup()

; ── Route B: Drop Unqualified Lead ──
same => n(hangup_call),NoOp(Lead dropped or not qualified. Hanging up channel.)
same => n,Hangup()
"""
    stdin, stdout, stderr = ssh.exec_command("cat > /etc/asterisk/extensions.conf")
    stdin.write(extensions_conf)
    stdin.close()
    print("extensions.conf written.")
    print("-" * 40)

    print("=== Reloading Asterisk Dialplan ===")
    run_cmd(ssh, "asterisk -rx 'dialplan reload'")
    run_cmd(ssh, "asterisk -rx 'dialplan show from-vicidial'")
    
    ssh.close()

if __name__ == "__main__":
    main()
