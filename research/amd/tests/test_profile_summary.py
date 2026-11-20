"""./profile's summary from rocprofv3 CSV files (made up here; no GPU)."""
import importlib.machinery
import importlib.util
import tempfile
import unittest
from pathlib import Path

AMD = Path(__file__).resolve().parents[1]
_loader = importlib.machinery.SourceFileLoader("ij_profile", str(AMD / "profile"))
_spec = importlib.util.spec_from_loader("ij_profile", _loader)
P = importlib.util.module_from_spec(_spec)
_loader.exec_module(P)

KT_HEAD = "Dispatch_Id,Kernel_Name,Start_Timestamp,End_Timestamp\n"
CC_HEAD = "Dispatch_Id,Kernel_Name,Counter_Name,Counter_Value\n"


def write(root: Path, rel: str, text: str):
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text)


class ProfileSummaryTest(unittest.TestCase):
    def make(self, root: Path):
        # pass 1: two dispatches of gemm, 1 ms each, 1 GiB read each: 2 x 1073741824 B / 0.002 s = 1073.7 GB/s
        write(root, "pass_1/run_kernel_trace.csv", KT_HEAD +
              "1,gemm,1000000000,1001000000\n2,gemm,1002000000,1003000000\n3,norm,1003000000,1004000000\n")
        write(root, "pass_1/run_counter_collection.csv", CC_HEAD +
              "1,gemm,FETCH_SIZE,1048576\n2,gemm,FETCH_SIZE,1048576\n"
              "1,gemm,MemUnitBusy,80\n2,gemm,MemUnitBusy,60\n"
              "3,norm,WRITE_SIZE,10\n9,norm,WRITE_SIZE,10\n")
        # pass 2: a counter whose unit is not known; no kernel trace there
        write(root, "pass_2/run_counter_collection.csv", CC_HEAD + "1,gemm,ALUStalledByLDS,30\n2,gemm,ALUStalledByLDS,50\n")

    def test_counters_per_pass_per_kernel_with_units(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.make(root)
            text = "\n".join(P.summarize(root, 12.5, [], True))
        self.assertIn("command wall time: 12.50 s", text)
        self.assertIn("counters, pass 1, pass_1", text)
        self.assertIn("counters, pass 2, pass_2", text)
        self.assertIn("FETCH_SIZE: 2.09715e+06 KiB in total; 1073.7 GB/s read", text)
        self.assertIn("MemUnitBusy: mean 70.0%, min 60.0%, max 80.0%", text)
        self.assertNotIn("140", text)                                    # 80% + 60% is never shown
        self.assertIn("WRITE_SIZE: 20 KiB in total; no GB/s: only 1 of 2 dispatches", text)
        self.assertIn("ALUStalledByLDS: per dispatch mean 40, min 30, max 50 (unit not known here, so not summed", text)
        self.assertNotIn("80 ", text.split("ALUStalledByLDS")[1])

    def test_wall_and_kernel_span_are_separate(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t)
            self.make(root)
            lines = P.summarize(root, 12.5, [], False)
        span = [x for x in lines if x.startswith("kernel span")][0]
        busy = [x for x in lines if x.startswith("GPU busy")][0]
        self.assertIn("0.004 s", span)
        self.assertIn("of the kernel span", busy)
        self.assertNotIn("wall", busy)
        self.assertFalse(any(x.startswith("counters") for x in lines))

    def test_no_kernels_says_why(self):
        with tempfile.TemporaryDirectory() as t:
            text = "\n".join(P.summarize(Path(t), 3.0, [], True))
        self.assertIn("not inside containers", text)
        self.assertIn("counters: none recorded", text)

    def test_help_names_persistent_kernels_and_docker(self):
        self.assertIn("persistent kernel", P.__doc__)
        self.assertIn("one row", P.__doc__)
        self.assertIn("Docker container is not traced", P.__doc__.replace("\n", " "))
        self.assertIn("log file", P.__doc__)


if __name__ == "__main__":
    unittest.main()
