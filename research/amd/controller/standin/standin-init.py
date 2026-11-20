"""The stand-in GPU machine's entry point: the part of cloud-init the driver
relies on (ssh_keys.ed25519_*, write_files), then sshd with that key only."""
import base64
import os

import yaml

doc = yaml.safe_load(base64.b64decode(os.environ.get("USER_DATA", "")).decode() or "{}") or {}
k = doc.get("ssh_keys") or {}
if k.get("ed25519_private"):
    with open("/etc/ssh/ssh_host_ed25519_key", "w") as f:
        f.write(k["ed25519_private"])
    os.chmod("/etc/ssh/ssh_host_ed25519_key", 0o600)
    with open("/etc/ssh/ssh_host_ed25519_key.pub", "w") as f:
        f.write(k["ed25519_public"].strip() + "\n")
for w in doc.get("write_files") or []:
    os.makedirs(os.path.dirname(w["path"]), mode=0o700, exist_ok=True)
    with open(w["path"], "a" if w.get("append") else "w") as f:
        f.write(w["content"])
    os.chmod(w["path"], int(w.get("permissions", "0644"), 8))
os.execv("/usr/sbin/sshd", ["/usr/sbin/sshd", "-D", "-e", "-o", "HostKey=/etc/ssh/ssh_host_ed25519_key",
                            "-o", "PermitRootLogin=prohibit-password", "-o", "PasswordAuthentication=no"])
