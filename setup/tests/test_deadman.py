"""tools/tpu_deadman.sh, the spend guard that runs on the TPU VM.

The interface is its command line (arm, status, extend, disarm) plus the one
thing systemd runs (`_fire`).  gcloud, systemctl, shutdown and the metadata
server's curl are fakes that record their argv; the clock is moved with
IJ_DEADMAN_NOW.  The property that matters most is the last section: nothing
that cancels or postpones the guard can make it act.
"""
from __future__ import annotations

import os
import stat
import subprocess
from pathlib import Path

import pytest

from conftest import REPO

SCRIPT = REPO / "tools" / "tpu_deadman.sh"
T0 = 1_800_000_000                                   # 2027-01-15 08:00:00 UTC

FAKES = {
    "gcloud": r"""#!/bin/bash
echo "gcloud $*" >> "$FAKE/calls.log"
echo "$HOME ${CLOUDSDK_CONFIG:-unset}" >> "$FAKE/homes.log"
case "$*" in
  "auth list"*) [ -n "${FAKE_ACCOUNT-sa@proj.iam}" ] && echo "${FAKE_ACCOUNT-sa@proj.iam}"; exit 0 ;;
  *"tpu-vm describe"*) exit "${FAKE_DESCRIBE_RC:-0}" ;;
  *"tpu-vm delete"*)
    echo "$0" >> "$FAKE/gcloud-path.log"
    n=$(cat "$FAKE/delete-tries" 2>/dev/null || echo 0); echo $((n + 1)) > "$FAKE/delete-tries"
    rc=${FAKE_DELETE_RC:-0}                               # FAKE_DELETE_FAILS=N: the first N tries fail
    if [ -n "${FAKE_DELETE_FAILS:-}" ]; then if [ "$n" -lt "$FAKE_DELETE_FAILS" ]; then rc=1; else rc=0; fi; fi
    [ "$rc" = 0 ] || echo "ERROR: permission denied" >&2
    exit "$rc" ;;
esac
exit 0
""",
    "systemctl": r"""#!/bin/bash
echo "systemctl $*" >> "$FAKE/calls.log"
case "$1" in
  is-active) [ -n "${FAKE_TIMER_DOWN:-}" ] && { echo inactive; exit 3; }; echo active ;;
  disable|stop) exit "${FAKE_STOP_RC:-0}" ;;
esac
exit 0
""",
    "shutdown": r"""#!/bin/bash
echo "shutdown $*" >> "$FAKE/calls.log"
""",
    # GNU timeout (not on macOS): records the box, then runs the command -- or, for a
    # command named in FAKE_TIMEOUT_KILLS, reports that it ran out of time.
    "timeout": r"""#!/bin/bash
echo "timeout $*" >> "$FAKE/calls.log"
while [ $# -gt 0 ]; do case "$1" in -k) shift 2 ;; -*) shift ;; *) break ;; esac; done
shift
case " ${FAKE_TIMEOUT_KILLS:-} " in *" $(basename "$1") "*) exit 124 ;; esac
exec "$@"
""",
    "sleep": r"""#!/bin/bash
echo "sleep $*" >> "$FAKE/calls.log"
[ -z "${FAKE_SLEEP_DISARMS:-}" ] || rm -f "$IJ_DEADMAN_DIR/deadline"      # the operator disarms meanwhile
exit 0
""",
    "sync": r"""#!/bin/bash
echo "sync $*" >> "$FAKE/calls.log"
""",
    "mountpoint": r"""#!/bin/bash
grep -qx "${!#}" "$FAKE/mounts" 2>/dev/null
""",
    "curl": r"""#!/bin/bash
url=${!#}
echo "curl $url" >> "$FAKE/calls.log"
[ -n "${FAKE_NO_METADATA:-}" ] && exit 22
case "$url" in
  */attributes/tpu-env) printf "ACCELERATOR_TYPE: 'v6e-4'\nNODE_ID: 'gemma-node'\nZONE: 'europe-west4-a'\n" ;;
  */project/project-id) echo my-project ;;
  */service-accounts/default/scopes) echo "${FAKE_SCOPES-https://www.googleapis.com/auth/cloud-platform}" ;;
  *) exit 22 ;;
esac
""",
}


class Guard:
    def __init__(self, root: Path):
        self.root = root
        self.bin, self.fake = root / "bin", root / "fake"
        self.state, self.units = root / "state", root / "units"
        self.disk, self.box = root / "disk", root / "box"          # the cache disk; the VM's own paths
        for d in (self.bin, self.fake, self.units, self.disk):
            d.mkdir()
        for name, body in FAKES.items():
            p = self.bin / name
            p.write_text(body)
            p.chmod(p.stat().st_mode | stat.S_IXUSR)
        self.results = self.box / "opt" / "bench" / "results"
        self.research = self.box / "home" / "researcher" / "research"
        (self.results / "exp-1").mkdir(parents=True)
        (self.results / "exp-1" / "results.json").write_text('{"tps": 126}\n')
        (self.research / "__pycache__").mkdir(parents=True)
        (self.research / "__pycache__" / "x.pyc").write_text("")
        (self.research / "NOTES.md").write_text("what was tried\n")
        # calibration does not exist yet: a run that has not got that far
        self.sources = [self.results, self.box / "opt" / "bench" / "calibration", self.research]

    def run(self, *args: str, now: int = T0, **env) -> subprocess.CompletedProcess:
        e = {"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.root), "LC_ALL": "C", "FAKE": str(self.fake),
             "IJ_DEADMAN_DIR": str(self.state), "IJ_UNIT_DIR": str(self.units),
             "IJ_DEADMAN_SOURCES": " ".join(map(str, self.sources)),
             "IJ_METADATA_URL": "http://metadata.test/v1", "IJ_DEADMAN_NOW": str(now), **env}
        return subprocess.run(["/bin/bash", str(SCRIPT), *args], env=e, capture_output=True, text=True,
                              cwd=self.root, timeout=60)

    def fire(self, now: int, **env) -> subprocess.CompletedProcess:
        """What systemd does: runs the installed copy of the script, not the repo's."""
        return subprocess.run(["/bin/bash", str(self.state / "tpu_deadman.sh"), "_fire"],
                              env={"PATH": f"{self.bin}:/usr/bin:/bin", "HOME": str(self.root), "LC_ALL": "C",
                                   "FAKE": str(self.fake), "IJ_DEADMAN_DIR": str(self.state),
                                   "IJ_DEADMAN_SOURCES": " ".join(map(str, self.sources)),
                                   "IJ_UNIT_DIR": str(self.units), "IJ_DEADMAN_NOW": str(now), **env},
                              capture_output=True, text=True, cwd=self.root, timeout=60)

    def bootstrapped(self, mounted: bool = True) -> None:
        """What setup/bootstrap.sh leaves behind: the cache disk mounted, and its mount point on file."""
        self.state.mkdir(exist_ok=True)
        (self.state / "save-to").write_text(f"{self.disk}\n")
        if mounted:
            (self.fake / "mounts").write_text(f"{self.disk}\n")

    def saved(self) -> Path:
        """The one directory this VM saves into."""
        (d,) = (self.disk / "inference-judge-saved").iterdir()
        return d

    def log(self) -> str:
        return (self.state / "deadman.log").read_text()

    def calls(self, prefix: str = "") -> list[str]:
        p = self.fake / "calls.log"
        return [l for l in (p.read_text().splitlines() if p.exists() else []) if l.startswith(prefix)]

    def deletes(self) -> list[str]:
        return self.calls("gcloud compute tpus tpu-vm delete")

    def armed_for(self, hours: str = "2") -> None:
        r = self.run("arm", hours)
        assert r.returncode == 0, r.stdout + r.stderr


@pytest.fixture
def g(tmp_path: Path) -> Guard:
    return Guard(tmp_path)


# ---------------------------------------------------------------------- arm
def test_arm_installs_an_absolute_persistent_timer_and_the_deadline(g):
    r = g.run("arm", "2")
    assert r.returncode == 0, r.stdout + r.stderr
    timer = (g.units / "inference-judge-deadman.timer").read_text()
    assert "OnCalendar=2027-01-15 10:00:00 UTC" in timer          # absolute, so a reboot cannot push it out
    assert "Persistent=true" in timer and "WantedBy=timers.target" in timer
    assert (g.state / "deadline").read_text().strip() == str(T0 + 7200)
    service = (g.units / "inference-judge-deadman.service").read_text()
    assert f"ExecStart=/bin/bash {g.state}/tpu_deadman.sh _fire" in service
    assert "systemctl enable inference-judge-deadman.timer" in g.calls() and "systemctl restart inference-judge-deadman.timer" in g.calls()


def test_arm_takes_name_zone_and_project_from_the_metadata_server(g):
    g.armed_for()
    assert (g.state / "node").read_text().split() == ["NAME=gemma-node", "ZONE=europe-west4-a", "PROJECT=my-project"]
    assert "gcloud compute tpus tpu-vm describe gemma-node --zone europe-west4-a --project my-project " \
           "--format=value(name)" in g.calls()


def test_arm_with_explicit_node_needs_no_metadata_server(g):
    r = g.run("arm", "1", "--name", "n1", "--zone", "us-central2-b", "--project", "p1", FAKE_NO_METADATA="1")
    assert r.returncode == 0, r.stdout + r.stderr
    assert (g.state / "node").read_text().split() == ["NAME=n1", "ZONE=us-central2-b", "PROJECT=p1"]


def test_arm_without_a_node_name_says_what_to_pass(g):
    r = g.run("arm", "1", FAKE_NO_METADATA="1")
    assert r.returncode == 2 and "--name" in r.stderr
    assert not (g.units / "inference-judge-deadman.timer").exists() and not (g.state / "deadline").exists()


@pytest.mark.parametrize("hours", ["0", "-1", "abc", "", "1000", "1e3"])
def test_arm_refuses_a_deadline_that_makes_no_sense(g, hours):
    r = g.run("arm", hours)
    assert r.returncode == 2
    assert not (g.units / "inference-judge-deadman.timer").exists() and not (g.state / "deadline").exists()


def test_arm_without_a_working_account_is_loud_but_still_arms(g):
    r = g.run("arm", "2", FAKE_ACCOUNT="")
    assert r.returncode == 3
    assert "DELETION IS NOT CONFIRMED" in r.stderr and "no active gcloud account" in r.stderr
    assert (g.units / "inference-judge-deadman.timer").exists() and (g.state / "deadline").exists()      # the fallback still helps


def test_arm_where_the_node_cannot_be_read_is_loud(g):
    r = g.run("arm", "2", FAKE_DESCRIBE_RC="1")
    assert r.returncode == 3 and "cannot read node gemma-node" in r.stderr


def test_arm_where_the_vm_lacks_the_cloud_scope_is_loud(g):
    r = g.run("arm", "2", FAKE_SCOPES="https://www.googleapis.com/auth/devstorage.read_only")
    assert r.returncode == 3 and "scopes lack cloud-platform" in r.stderr


def test_arm_says_so_when_the_timer_does_not_come_up(g):
    r = g.run("arm", "2", FAKE_TIMER_DOWN="1")
    assert r.returncode == 2 and "NOTHING WILL DELETE THIS VM" in r.stderr


def test_arm_again_replaces_the_deadline(g):
    g.armed_for("5")
    g.run("arm", "1", now=T0 + 60)
    assert (g.state / "deadline").read_text().strip() == str(T0 + 60 + 3600)


# ------------------------------------------------------------ status, extend
def test_status_reports_the_node_and_the_time_left(g):
    g.armed_for("2")
    r = g.run("status", now=T0 + 1800)
    assert r.returncode == 0
    assert "gemma-node" in r.stdout and "2027-01-15 10:00:00 UTC" in r.stdout and "time left: 1h30m" in r.stdout
    assert "timer: active" in r.stdout


def test_status_when_nothing_is_armed_says_nothing_will_delete(g):
    r = g.run("status")
    assert r.returncode == 1 and "not armed" in r.stdout


def test_status_warns_when_the_timer_is_dead(g):
    g.armed_for()
    r = g.run("status", FAKE_TIMER_DOWN="1")
    assert r.returncode == 4 and "nothing will fire" in r.stderr


def test_extend_adds_to_the_deadline_and_moves_the_timer(g):
    g.armed_for("2")
    r = g.run("extend", "1.5", now=T0 + 600)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (g.state / "deadline").read_text().strip() == str(T0 + 7200 + 5400)
    assert "OnCalendar=2027-01-15 11:30:00 UTC" in (g.units / "inference-judge-deadman.timer").read_text()


def test_extend_when_not_armed_or_already_late_refuses(g):
    assert g.run("extend", "1").returncode == 2
    g.armed_for("1")
    r = g.run("extend", "1", now=T0 + 7200)
    assert r.returncode == 2 and "has passed" in r.stderr
    assert (g.state / "deadline").read_text().strip() == str(T0 + 3600)


# -------------------------------------------------------------------- disarm
def test_disarm_removes_the_deadline_the_timer_and_the_units(g):
    g.armed_for()
    r = g.run("disarm")
    assert r.returncode == 0
    assert not (g.state / "deadline").exists()
    assert not (g.units / "inference-judge-deadman.timer").exists() and not (g.units / "inference-judge-deadman.service").exists()
    assert "systemctl disable --now inference-judge-deadman.timer" in g.calls()
    assert g.run("status").returncode == 1


# --------------------------------------------------------------------- fire
def test_at_the_deadline_it_deletes_this_node(g):
    g.armed_for("2")
    r = g.fire(T0 + 7200)
    assert r.returncode == 0, r.stdout + r.stderr
    assert g.deletes() == ["gcloud compute tpus tpu-vm delete gemma-node --zone europe-west4-a "
                           "--project my-project --quiet"]
    assert not g.calls("shutdown")
    assert "DEADLINE REACHED" in (g.state / "deadman.log").read_text()


def test_a_failed_delete_is_retried_then_falls_back_to_shutdown_and_says_the_node_may_still_bill(g):
    g.armed_for("2")
    r = g.fire(T0 + 7300, FAKE_DELETE_RC="1")
    assert len(g.deletes()) == 5                                     # one refusal is not the end
    assert g.calls("sleep") == ["sleep 30", "sleep 60", "sleep 120", "sleep 240"]
    assert g.calls("shutdown") == ["shutdown -h now"]
    log = (g.state / "deadman.log").read_text()
    assert "DELETE FAILED (exit 1)" in log and "STILL BE BILLED" in log and "gemma-node" in log
    assert "permission denied" in log                                # gcloud's own words are kept
    assert r.returncode == 0
    assert (g.state / "deadline").exists()                           # still due: a restart tries again


def test_a_delete_that_works_on_the_third_try_stops_there(g):
    g.armed_for("2")
    g.fire(T0 + 7200, FAKE_DELETE_FAILS="2")
    assert len(g.deletes()) == 3 and g.calls("sleep") == ["sleep 30", "sleep 60"]
    assert not g.calls("shutdown") and not (g.state / "deadline").exists()


def test_a_vm_restarted_after_a_failed_delete_fires_again(g):
    g.armed_for("2")
    g.fire(T0 + 7200, FAKE_DELETE_RC="1")                            # halted, still billed
    service = (g.units / "inference-judge-deadman.service").read_text()
    assert "WantedBy=multi-user.target" in service                   # a boot runs it; the past timer would not
    assert "systemctl enable inference-judge-deadman.service" in g.calls()
    r = g.fire(T0 + 7200 + 86400)                                    # someone starts the VM a day later
    assert r.returncode == 0 and len(g.deletes()) == 6
    assert not (g.state / "deadline").exists()


def test_the_delete_runs_the_gcloud_arm_found_whatever_path_the_unit_has(g):
    """systemd's PATH is not the operator's: gcloud from a snap or /usr/local is not on it."""
    g.armed_for("2")
    units_path = g.root / "systemd-path"
    units_path.mkdir()
    for name in FAKES:
        if name != "gcloud":
            (units_path / name).symlink_to(g.bin / name)
    r = g.fire(T0 + 7200, PATH=f"{units_path}:/usr/bin:/bin")
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(g.deletes()) == 1 and not g.calls("shutdown")
    assert (g.fake / "gcloud-path.log").read_text().split() == [str(g.bin / "gcloud")]


def test_disarming_while_the_delete_is_being_retried_stops_it(g):
    g.armed_for("2")
    r = g.fire(T0 + 7200, FAKE_DELETE_RC="1", FAKE_SLEEP_DISARMS="1")
    assert len(g.deletes()) == 1 and not g.calls("shutdown")
    assert "disarmed" in r.stdout


def test_firing_twice_deletes_once(g):
    g.armed_for("2")
    g.fire(T0 + 7200)
    g.fire(T0 + 7210)
    assert len(g.deletes()) == 1


# ----------------------------------------- cancelling can never fire the action
def test_a_timer_that_fires_early_does_nothing_and_re_arms(g):
    g.armed_for("2")
    n = len(g.calls("systemctl restart"))
    r = g.fire(T0 + 3600)                                            # an hour early
    assert r.returncode == 0 and "refusing" in r.stdout
    assert not g.deletes() and not g.calls("shutdown")
    assert len(g.calls("systemctl restart")) == n + 1


def test_a_timer_that_outlives_disarm_does_nothing(g):
    """disarm could not stop the timer (systemctl fails): the fire script still finds no deadline."""
    g.armed_for("2")
    installed = g.state / "tpu_deadman.sh"
    g.run("disarm", FAKE_STOP_RC="1")
    assert installed.exists()                                        # the script a stale unit would still run
    r = g.fire(T0 + 7200 + 3600)                                     # long past the old deadline
    assert r.returncode == 0 and "no deadline" in r.stdout
    assert not g.deletes() and not g.calls("shutdown")


def test_a_manual_start_of_the_service_before_the_deadline_does_nothing(g):
    g.armed_for("2")
    g.fire(T0 + 5)
    g.fire(T0 + 7199)
    assert not g.deletes() and not g.calls("shutdown")


def test_the_old_timer_firing_after_an_extend_does_nothing(g):
    g.armed_for("2")
    g.run("extend", "1", now=T0 + 100)
    r = g.fire(T0 + 7200)                                            # the original deadline: stale timer
    assert r.returncode == 0 and "refusing" in r.stdout
    assert not g.deletes()
    g.fire(T0 + 7200 + 3600)                                         # the new deadline
    assert len(g.deletes()) == 1


def test_a_deadline_file_that_is_missing_stops_everything_even_when_the_node_is_known(g):
    g.armed_for("1")
    (g.state / "deadline").unlink()
    g.fire(T0 + 99999)
    assert not g.deletes() and not g.calls("shutdown")


def test_the_installed_copy_survives_deleting_the_repo_checkout(g):
    g.armed_for("1")
    assert os.access(g.state / "tpu_deadman.sh", os.X_OK)
    assert (g.state / "tpu_deadman.sh").read_text() == SCRIPT.read_text()


def test_the_account_arm_checks_is_the_account_the_deadline_uses(g):
    """arm runs under sudo with whatever HOME it likes; the unit runs as root.  Both pin one home."""
    homes = g.root / "pinned-home"
    g.run("arm", "1", IJ_DEADMAN_HOME=str(homes), HOME=str(g.root / "someone-elses"), CLOUDSDK_CONFIG="/tmp/x")
    g.fire(T0 + 3600, IJ_DEADMAN_HOME=str(homes), HOME="/root")
    seen = (g.fake / "homes.log").read_text().splitlines()
    assert len(seen) >= 3 and set(seen) == {f"{homes} unset"}


def test_the_unit_runs_as_the_home_the_checks_used(g):
    g.armed_for()
    assert "Environment=HOME=/root" in (g.units / "inference-judge-deadman.service").read_text()


# ------------------------------------------ the results outlive the VM
def test_before_the_delete_the_results_are_copied_to_the_cache_disk(g):
    g.bootstrapped()
    g.armed_for("2")
    r = g.fire(T0 + 7200)
    assert r.returncode == 0, r.stdout + r.stderr
    d = g.saved()
    assert d.name == "gemma-node-20270115T080000Z"                   # this VM's own directory
    assert (d / str(g.results).lstrip("/") / "exp-1" / "results.json").read_text() == '{"tps": 126}\n'
    assert (d / str(g.research).lstrip("/") / "NOTES.md").exists()
    assert not list(d.rglob("*.pyc"))
    assert (g.disk / "inference-judge-saved").stat().st_mode & 0o077 == 0      # the researcher cannot read or plant there
    calls = g.calls()
    copies = [i for i, l in enumerate(calls) if l.startswith("timeout") and " rsync " in l]
    flushed = calls.index("sync ")
    deleted = next(i for i, l in enumerate(calls) if l.startswith("gcloud compute tpus tpu-vm delete"))
    assert len(copies) == 2 and max(copies) < flushed < deleted      # copied, flushed to the disk, THEN deleted
    assert f"saved {g.results}" in g.log() and f"saved {g.research}" in g.log()


def test_the_copy_and_the_flush_are_time_boxed(g):
    g.bootstrapped()
    g.armed_for("2")
    g.fire(T0 + 7200, IJ_DEADMAN_SAVE_S="77")
    boxed = [l.split() for l in g.calls("timeout") if " rsync " in l or l.endswith(" sync")]
    assert len(boxed) == 3
    for argv in boxed:                                               # timeout -k 10 SECONDS command...
        assert argv[1:3] == ["-k", "10"] and 0 < int(argv[3]) <= 120, argv
    assert all(int(a[3]) <= 77 for a in boxed if a[4] == "rsync")


def test_a_copy_that_runs_out_of_time_is_logged_and_the_vm_is_still_deleted(g):
    g.bootstrapped()
    g.armed_for("2")
    r = g.fire(T0 + 7200, FAKE_TIMEOUT_KILLS="rsync")
    assert r.returncode == 0 and len(g.deletes()) == 1
    assert f"NOT SAVED {g.results}: ran out of time" in g.log()
    assert f"NOT SAVED {g.research}: ran out of time" in g.log()
    assert "RESULTS NOT FULLY SAVED" in g.log()


def test_without_a_cache_disk_on_file_it_says_the_results_die_with_the_vm_and_still_deletes(g):
    g.armed_for("2")                                                 # bootstrap never ran
    r = g.fire(T0 + 7200)
    assert r.returncode == 0 and len(g.deletes()) == 1
    assert "RESULTS NOT SAVED" in g.log() and "save-to" in g.log()
    assert not g.calls("timeout -k 10 600 rsync")


def test_a_disk_that_is_not_mounted_is_never_written_to(g):
    """The mount point is then a directory on the boot disk: a copy there dies with the VM."""
    g.bootstrapped(mounted=False)
    g.armed_for("2")
    g.fire(T0 + 7200)
    assert "is not a mounted disk" in g.log() and "RESULTS NOT SAVED" in g.log()
    assert not (g.disk / "inference-judge-saved").exists() and len(g.deletes()) == 1


def test_a_source_the_run_has_not_made_yet_is_noted_and_is_not_a_failure(g):
    g.bootstrapped()
    g.armed_for("2")
    r = g.run("save")
    assert r.returncode == 0, r.stdout + r.stderr
    assert f"{g.sources[1]} is not there" in r.stdout and "NOT SAVED" not in r.stdout


def test_arm_schedules_an_hourly_save_and_disarm_removes_it(g):
    g.armed_for("2")
    timer = (g.units / "inference-judge-deadman-save.timer").read_text()
    assert "OnCalendar=hourly" in timer and "WantedBy=timers.target" in timer
    assert f"ExecStart=/bin/bash {g.state}/tpu_deadman.sh save" in (g.units / "inference-judge-deadman-save.service").read_text()
    assert "systemctl enable inference-judge-deadman-save.timer" in g.calls()
    g.run("disarm")
    assert not (g.units / "inference-judge-deadman-save.timer").exists()
    assert not (g.units / "inference-judge-deadman-save.service").exists()
    assert "systemctl disable --now inference-judge-deadman-save.timer" in g.calls()


def test_the_hourly_save_is_stopped_before_the_last_one_starts(g):
    g.bootstrapped()
    g.armed_for("2")
    g.fire(T0 + 7200)
    calls = g.calls()
    stop = calls.index("systemctl stop inference-judge-deadman-save.timer inference-judge-deadman-save.service")
    assert stop < next(i for i, l in enumerate(calls) if " rsync " in l)


def test_each_save_adds_to_the_last_and_never_removes(g):
    g.bootstrapped()
    g.armed_for("5")
    assert g.run("save", now=T0 + 3600).returncode == 0
    (g.results / "exp-1" / "results.json").unlink()                  # a run directory cleaned up on the VM
    (g.results / "exp-2").mkdir()
    (g.results / "exp-2" / "results.json").write_text("{}\n")
    assert g.run("save", now=T0 + 7200).returncode == 0
    kept = g.saved() / str(g.results).lstrip("/")
    assert (kept / "exp-1" / "results.json").exists() and (kept / "exp-2" / "results.json").exists()


def test_a_later_vm_with_the_same_name_saves_beside_the_first_not_over_it(g, tmp_path_factory):
    g.bootstrapped()
    g.armed_for("2")
    g.run("save")
    later = Guard(tmp_path_factory.mktemp("later-vm"))               # a fresh boot disk, the same cache disk
    later.disk = g.disk
    later.bootstrapped()
    assert later.run("arm", "2", now=T0 + 86400).returncode == 0
    assert later.run("save", now=T0 + 90000).returncode == 0
    assert sorted(p.name for p in (g.disk / "inference-judge-saved").iterdir()) == [
        "gemma-node-20270115T080000Z", "gemma-node-20270116T080000Z"]


def test_status_says_where_the_results_go_and_when_they_last_went(g):
    g.armed_for("2")
    r = g.run("status")
    assert "results: NOT SAVED" in r.stdout                          # no disk on file: say so before it matters
    g.bootstrapped()
    g.run("save", now=T0 + 600)
    r = g.run("status", now=T0 + 900)
    assert f"results: 2027-01-15 08:10:00 UTC: 2 saved, 0 not saved, in {g.saved()}" in r.stdout
