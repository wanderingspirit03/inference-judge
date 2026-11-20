#!/usr/bin/env python3
"""A fake DigitalOcean API for the controller's tests: the droplet calls the
driver makes, answered with Docker containers (image ij-standin-box) on this
machine.  Run as root (it uses Docker).  Stdlib.

  fake_do.py <port> <token file> <price per hour>

  GET    /v2/sizes                      251 sizes, 200 a page at most; gpu-mi350x1-288gb-spot (the
                                        given price) is the last, on page 2 (links.pages.next)
  GET    /v2/account/keys               the ssh keys registered here (id, name, public_key, fingerprint)
  POST   /v2/account/keys               register one {name, public_key}
  GET    /v2/droplets?tag_name=T        droplets with the tag
  POST   /v2/droplets                   start a container; user_data goes in as $USER_DATA; 422 without
                                        ssh_keys, as DigitalOcean answers for an image without root passwords
  GET    /v2/droplets/<id>              status new, then active with the container's address
  DELETE /v2/droplets/<id>              remove the container
  POST   /fake/reclaim/<id>             (tests) remove it as a spot reclaim would
  POST   /fake/fail-delete/<n>          (tests) the next n DELETEs answer 500
  GET    /fake/calls                    (tests) every call so far, with the
                                        number of droplets that existed then
Every /v2 call needs "Authorization: Bearer <the token in the file>"."""
import base64
import hashlib
import json
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlsplit

DROPLETS: dict[int, dict] = {}
CALLS: list = []
FAIL_DELETE = [0]
LOCK = threading.Lock()
TOKEN = ""
PRICE = 1.0
NEXT = [500000001]


KEYS: list = []
NEXT_KEY = [7000001]


def fingerprint(pub: str) -> str:
    h = hashlib.md5(base64.b64decode(pub.split()[1])).hexdigest()
    return ":".join(h[i:i + 2] for i in range(0, 32, 2))


def page(u, key: str, items: list) -> dict:
    """DigitalOcean's paging: per_page at most 200 (default 20), links.pages.next a full URL."""
    q = parse_qs(u.query)
    per = min(int(q.get("per_page", ["20"])[0]), 200)
    n = int(q.get("page", ["1"])[0])
    out = {key: items[(n - 1) * per:n * per], "links": {"pages": {}}, "meta": {"total": len(items)}}
    if n * per < len(items):
        rest = "&".join(f"{k}={v[0]}" for k, v in q.items() if k not in ("page", "per_page"))
        out["links"]["pages"]["next"] = (f"https://api.digitalocean.com{u.path}?page={n + 1}&per_page={per}"
                                         + (f"&{rest}" if rest else ""))
    return out


def docker(*a) -> str:
    return subprocess.run(["docker", *a], capture_output=True, text=True).stdout.strip()


def view(d: dict) -> dict:
    v = {k: d[k] for k in ("id", "name", "tags", "size_slug", "created_at")}
    v["region"] = {"slug": d["region"]}
    if time.time() - d["t"] < 3:
        v["status"], v["networks"] = "new", {"v4": []}
    else:
        ip = docker("inspect", "-f", "{{if .State.Running}}{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}{{end}}",
                    d["container"])
        if ip:
            v["status"], v["networks"] = "active", {"v4": [{"type": "public", "ip_address": ip}]}
        else:
            v["status"], v["networks"] = "off", {"v4": []}
    return v


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj=None):
        raw = b"" if obj is None else json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _auth(self) -> bool:
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._json(401, {"id": "unauthorized", "message": "Unable to authenticate you"})
            return False
        return True

    def _route(self, method):
        u = urlsplit(self.path)
        parts = u.path.strip("/").split("/")
        body = None
        if self.headers.get("Content-Length"):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"null")
        with LOCK:
            CALLS.append({"t": time.time(), "method": method, "path": u.path, "live": len(DROPLETS),
                          "body": {k: v for k, v in (body or {}).items() if k != "user_data"} if isinstance(body, dict) else None})
        if parts[0] == "fake":
            if parts[1] == "calls":
                return self._json(200, CALLS)
            if parts[1] == "fail-delete" and method == "POST":
                FAIL_DELETE[0] = int(parts[2])
                return self._json(200, {})
            if parts[1] == "reclaim" and method == "POST":
                d = DROPLETS.pop(int(parts[2]), None)
                if d:
                    docker("rm", "-f", d["container"])
                return self._json(200 if d else 404, {})
            return self._json(404, {})
        if not self._auth():
            return
        if parts == ["v2", "sizes"]:
            # as DigitalOcean: many sizes, at most 200 a page; the GPU size is not on the first page
            sizes = [{"slug": f"s-filler-{k}", "price_hourly": 0.01} for k in range(250)]
            return self._json(200, page(u, "sizes", sizes + [{"slug": "gpu-mi350x1-288gb-spot", "price_hourly": PRICE}]))
        if parts == ["v2", "account", "keys"] and method == "GET":
            return self._json(200, page(u, "ssh_keys", KEYS))
        if parts == ["v2", "account", "keys"] and method == "POST":
            k = {"id": NEXT_KEY[0], "name": body["name"], "public_key": body["public_key"],
                 "fingerprint": fingerprint(body["public_key"])}
            NEXT_KEY[0] += 1
            KEYS.append(k)
            return self._json(201, {"ssh_key": k})
        if parts == ["v2", "droplets"] and method == "GET":
            tag = parse_qs(u.query).get("tag_name", [None])[0]
            return self._json(200, page(u, "droplets", [view(d) for d in DROPLETS.values() if tag is None or tag in d["tags"]]))
        if parts == ["v2", "droplets"] and method == "POST" and not body.get("ssh_keys"):
            return self._json(422, {"id": "unprocessable_entity",
                                    "message": "The image for this droplet does not use root passwords, please use an SSH key"})
        if parts == ["v2", "droplets"] and method == "POST":
            i = NEXT[0]
            NEXT[0] += 1
            name = f"ij-fake-{i}"
            ud = base64.b64encode(body.get("user_data", "").encode()).decode()
            cid = docker("run", "-d", "--name", name, "--label", "ij.fake=1", "--cap-add", "NET_ADMIN", "-e", f"USER_DATA={ud}", "ij-standin-box")
            if not cid:
                return self._json(500, {"message": "docker run failed"})
            DROPLETS[i] = {"id": i, "name": body["name"], "tags": body.get("tags") or [], "size_slug": body["size"],
                           "region": body["region"], "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                           "t": time.time(), "container": name}
            return self._json(202, {"droplet": view(DROPLETS[i])})
        if len(parts) == 3 and parts[:2] == ["v2", "droplets"]:
            d = DROPLETS.get(int(parts[2]))
            if d is None:
                return self._json(404, {"id": "not_found"})
            if method == "GET":
                return self._json(200, {"droplet": view(d)})
            if method == "DELETE":
                if FAIL_DELETE[0] > 0:
                    FAIL_DELETE[0] -= 1
                    return self._json(500, {"id": "server_error", "message": "fake failure"})
                DROPLETS.pop(d["id"])
                docker("rm", "-f", d["container"])
                return self._json(204)
        self._json(404, {"id": "not_found"})

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def do_DELETE(self):
        self._route("DELETE")


def main(argv):
    global TOKEN, PRICE
    TOKEN = open(argv[1]).read().strip()
    PRICE = float(argv[2])
    ThreadingHTTPServer(("127.0.0.1", int(argv[0])), H).serve_forever()


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
