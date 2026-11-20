"""examples/atom-engine/shim.py with SHIM_IDS_ONLY=1 (AMD's tuned engine:
token ids, no logprobs) against a fake server that answers with token ids.

Needs numpy and tokenizers (bench's environment):
  cd bench && uv run --frozen python -m unittest discover -s ../research/amd/tests -p 'test_shim_*.py'
"""
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
try:
    import numpy as np  # noqa: F401
    import tokenizers  # noqa: F401
except ImportError:  # pragma: no cover
    raise unittest.SkipTest("needs numpy and tokenizers (run it in bench's uv environment)")
sys.path.append(str(AMD.parents[1] / "bench" / "tests"))
import fake_atom  # noqa: E402

SHIM = AMD / "examples" / "atom-engine" / "shim.py"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class IdsOnlyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        fake_atom.train_tokenizer(str(Path(self.tmp.name) / "tokenizer.json"))
        self.seen, self.reply = [], {}
        test = self

        class H(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                test.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
                b = json.dumps(test.reply).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)

        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.port = free_port()
        env = {**os.environ, "SHIM_IDS_ONLY": "1", "IJ_REF_MODEL_DIR": self.tmp.name}
        self.p = subprocess.Popen([sys.executable, str(SHIM), str(self.port),
                                   f"http://127.0.0.1:{self.srv.server_address[1]}"], env=env,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", self.port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)

    def tearDown(self):
        self.p.kill()
        self.p.wait()
        self.srv.shutdown()
        self.srv.server_close()
        self.tmp.cleanup()

    def ask(self) -> tuple[int, dict]:
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/v1/completions", method="POST",
                                     data=json.dumps({"prompt": [1, 2, 3], "max_tokens": 4, "logprobs": 5}).encode())
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read())

    def test_token_ids_are_passed_on_without_logprobs(self):
        self.reply = {"id": "x", "choices": [{"finish_reason": "length", "token_ids": [7, 8, 9, 10],
                                              "message": {"content": "abcd"}}], "usage": {"completion_tokens": 4}}
        st, doc = self.ask()
        self.assertEqual(st, 200)
        ch = doc["choices"][0]
        self.assertEqual(ch["token_ids"], [7, 8, 9, 10])
        self.assertNotIn("logprobs", ch)
        self.assertTrue(self.seen[0].get("return_token_ids"))
        self.assertNotIn("logprobs", self.seen[0])

    def test_ids_in_the_message_are_found_too(self):
        self.reply = {"choices": [{"finish_reason": "length", "message": {"content": "ab", "token_ids": [5, 6]}}]}
        st, doc = self.ask()
        self.assertEqual((st, doc["choices"][0]["token_ids"]), (200, [5, 6]))

    def test_no_ids_fails_closed(self):
        self.reply = {"choices": [{"finish_reason": "length", "message": {"content": "abcd"}}]}
        st, doc = self.ask()
        self.assertEqual(st, 502)
        self.assertIn("no token ids", doc["error"])


if __name__ == "__main__":
    unittest.main()
