"""controller/driver/ij-driver against the fake DigitalOcean (standin/fake_do.py,
in this process, no Docker): the size's price on a later page, the driver's
own ssh key registered or reused, and a refused create that puts the state
back to none.  The two faults of the first real creation (2026-11-18)."""
import importlib.machinery
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

AMD = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(AMD / "controller" / "lib"))
sys.path.insert(0, str(AMD / "controller" / "standin"))
import fake_do  # noqa: E402
import ijctl    # noqa: E402

HAVE_KEYGEN = shutil.which("ssh-keygen") is not None


class FakeDOBase(unittest.TestCase):
    def setUp(self):
        self.td = Path(tempfile.mkdtemp())
        (self.td / "ctl" / "ssh").mkdir(parents=True)
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-C", "driver", "-f",
                        str(self.td / "ctl" / "ssh" / "driver")], check=True)
        tok = self.td / "do-token"
        tok.write_text("t0ken\n")
        tok.chmod(0o600)
        fake_do.TOKEN, fake_do.PRICE = "t0ken", 3.0
        fake_do.KEYS.clear()
        fake_do.CALLS.clear()
        fake_do.DROPLETS.clear()
        self.srv = ThreadingHTTPServer(("127.0.0.1", 0), fake_do.H)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.cfg = json.loads((AMD / "controller" / "controller.installed.json").read_text())
        self.cfg["paths"]["ctl_dir"] = str(self.td / "ctl")
        self.cfg["agent"]["enabled"] = False
        self.cfg["driver"].update({"api_base": f"http://127.0.0.1:{self.srv.server_address[1]}",
                                   "do_token_file": str(tok), "price_usd_per_hour": None, "do_ssh_key_ids": [],
                                   "ledger": str(self.td / "ctl" / "machines.jsonl")})

    def tearDown(self):
        self.srv.shutdown()
        self.srv.server_close()
        shutil.rmtree(self.td, ignore_errors=True)

    def driver(self):
        p = self.td / "cfg.json"
        p.write_text(json.dumps(self.cfg))
        ijctl.CONFIG = p
        loader = importlib.machinery.SourceFileLoader("ij_driver", str(AMD / "controller" / "driver" / "ij-driver"))
        spec = importlib.util.spec_from_loader("ij_driver", loader)
        m = importlib.util.module_from_spec(spec)
        loader.exec_module(m)
        return m

    def calls(self, method, path):
        return [c for c in fake_do.CALLS if c["method"] == method and c["path"] == path]


@unittest.skipUnless(HAVE_KEYGEN, "needs ssh-keygen")
class DriverAgainstFakeDO(FakeDOBase):

    def test_the_price_is_found_on_a_later_page(self):
        d = self.driver()
        self.assertEqual(d.price(), 3.0)
        self.assertEqual(len(self.calls("GET", "/v2/sizes")), 2)       # 251 sizes, 200 a page

    def test_a_set_price_is_used_without_asking(self):
        self.cfg["driver"]["price_usd_per_hour"] = 2.5
        self.assertEqual(self.driver().price(), 2.5)
        self.assertEqual(self.calls("GET", "/v2/sizes"), [])

    def test_the_drivers_key_is_registered_once_then_reused(self):
        d = self.driver()
        pub = (self.td / "ctl" / "ssh" / "driver.pub").read_text().strip()
        ids = d.ssh_key_ids()
        self.assertEqual(len(fake_do.KEYS), 1)
        self.assertEqual(fake_do.KEYS[0]["public_key"], pub)              # the public key only
        self.assertEqual(ids, [fake_do.KEYS[0]["id"]])
        self.assertEqual(fake_do.KEYS[0]["fingerprint"], d.fingerprint(pub))
        self.assertEqual(d.ssh_key_ids(), ids)
        self.assertEqual(len(self.calls("POST", "/v2/account/keys")), 1)

    def test_configured_key_ids_are_used_as_they_are(self):
        self.cfg["driver"]["do_ssh_key_ids"] = [42]
        self.assertEqual(self.driver().ssh_key_ids(), [42])
        self.assertEqual(self.calls("GET", "/v2/account/keys"), [])

    def test_a_refused_create_puts_the_state_back_to_none_with_the_reason(self):
        self.cfg["driver"]["do_ssh_key_ids"] = []
        d = self.driver()
        d.ssh_key_ids = lambda: []                       # as before the fix: no key, DigitalOcean answers 422
        self.assertEqual(d.tick(), 0)
        st = ijctl.box_state(self.cfg)
        self.assertEqual(st["state"], "none")
        self.assertIn("did not create", st["why"])
        self.assertIn("422", st["why"])
        self.assertEqual(d.open_machines(), [])


@unittest.skipUnless(HAVE_KEYGEN, "needs ssh-keygen")
class DriverRestartedDuringAChange(FakeDOBase):
    """A driver stopped in the middle of create() (a restart, 2026-11-18) left
    box.json at "creating" with an open machine nothing set up.  reconcile()
    resumes the set-up of such a machine when the host key pinned before the
    droplet was made is the one made for it, and destroys it otherwise."""

    def setUp(self):
        super().setUp()
        self.real_docker = fake_do.docker
        fake_do.docker = lambda *a: "10.9.9.9" if a[0] == "inspect" else "cid"     # no Docker in unit tests
        self.cfg["driver"]["do_ssh_key_ids"] = [7]
        self.cfg["driver"]["price_usd_per_hour"] = 3.0
        self.cfg["driver"]["spend_cap_usd"] = 250

    def tearDown(self):
        fake_do.docker = self.real_docker
        super().tearDown()

    def stopped_after_the_droplet_was_made(self, d, in_ledger=True):
        """create() up to the POST, then the driver is gone (set_up never ran)."""
        calls = []
        d.set_up = lambda did, name: calls.append((did, name))
        d.create()
        self.assertEqual(len(calls), 1)
        did, name = calls[0]
        for x in fake_do.DROPLETS.values():
            x["t"] -= 10                                         # active now
        if not in_ledger:                                        # stopped between the POST and the ledger line
            d.LEDGER.write_text("")
        return did, name

    def resumed(self, d):
        calls = []
        d.set_up = lambda did, name: calls.append((did, name))
        d.reconcile()
        return calls

    def test_the_host_key_is_pinned_with_the_name_before_the_droplet_is_made(self):
        d = self.driver()
        posts = []
        real_api = d.api

        def api(method, path, body=None):
            if method == "POST" and path == "/v2/droplets":
                pin = json.loads((self.td / "ctl" / "box_pin.json").read_text())
                posts.append((pin["name"] == body["name"], pin["host_key"].split()[1] in body["user_data"]))
            return real_api(method, path, body)
        d.api = api
        did, name = self.stopped_after_the_droplet_was_made(d)
        self.assertEqual(posts, [(True, True)])
        kh = (self.td / "ctl" / "box_known_hosts").read_text()
        self.assertTrue(kh.startswith("ij-box ssh-ed25519 "))
        st = ijctl.box_state(self.cfg)
        self.assertEqual((st["state"], st["id"], st["name"]), ("creating", did, name))

    def test_a_restart_after_the_create_resumes_the_set_up(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver())
        d = self.driver()                                        # the restarted driver
        self.assertEqual(self.resumed(d), [(did, name)])
        self.assertEqual(self.calls("DELETE", f"/v2/droplets/{did}"), [])
        self.assertEqual(len(d.open_machines()), 1)

    def test_a_restart_in_setting_up_resumes_too(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver())
        ijctl.write_box_state(self.cfg, state="setting-up", id=did, name=name, ip="10.9.9.9")
        self.assertEqual(self.resumed(self.driver()), [(did, name)])

    def test_a_droplet_the_ledger_missed_is_adopted_and_set_up_when_the_pin_is_its_own(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver(), in_ledger=False)
        d = self.driver()
        self.assertEqual(self.resumed(d), [(did, name)])
        self.assertTrue(d.machines()[str(did)].get("adopted"))

    def test_without_its_own_host_key_the_machine_is_destroyed_not_used(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver())
        (self.td / "ctl" / "box_pin.json").write_text(json.dumps({"name": "ij-amd-gpu-older", "host_key": "ssh-ed25519 AAAA x"}))
        d = self.driver()
        self.assertEqual(self.resumed(d), [])
        self.assertEqual(len(self.calls("DELETE", f"/v2/droplets/{did}")), 1)
        self.assertEqual(d.open_machines(), [])
        self.assertEqual(ijctl.box_state(self.cfg)["state"], "none")
        (self.td / "ctl" / "box_pin.json").unlink()

    def test_a_restart_while_destroying_finishes_the_destruction(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver())
        ijctl.write_box_state(self.cfg, state="destroying", why="spending cap")
        d = self.driver()
        self.assertEqual(self.resumed(d), [])
        self.assertEqual(len(self.calls("DELETE", f"/v2/droplets/{did}")), 1)
        self.assertEqual(ijctl.box_state(self.cfg)["state"], "none")

    def test_a_ready_machine_is_left_alone(self):
        did, name = self.stopped_after_the_droplet_was_made(self.driver())
        ijctl.write_box_state(self.cfg, state="ready", id=did, name=name, ip="10.9.9.9")
        (self.td / "ctl" / "box_pin.json").unlink()              # a machine made before the pin existed
        d = self.driver()
        self.assertEqual(self.resumed(d), [])
        self.assertEqual(self.calls("DELETE", f"/v2/droplets/{did}"), [])
        self.assertEqual(ijctl.box_state(self.cfg)["state"], "ready")


if __name__ == "__main__":
    unittest.main()
