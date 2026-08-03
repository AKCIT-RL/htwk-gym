"""Temporary standalone runner: executes the tests/unit/test_*.py modules inside
the mimickit container image, which lacks pip/pytest. Provides a minimal pytest
shim."""
import glob, importlib, inspect, os, re, sys, types, contextlib, traceback


class _Approx:
    def __init__(self, expected, rel=None, abs=None):
        self.e, self.rel, self.abs = expected, rel, abs

    def __eq__(self, other):
        tol = self.abs if self.abs is not None else (self.rel or 1e-6) * max(1e-12, abs(self.e))
        return abs(other - self.e) <= tol


pytest = types.ModuleType("pytest")
pytest.approx = lambda e, rel=None, abs=None: _Approx(e, rel, abs)


@contextlib.contextmanager
def _raises(exc, match=None):
    try:
        yield
    except exc as e:
        if match is not None and not re.search(match, str(e)):
            raise AssertionError("pattern {!r} not found in {!r}".format(match, str(e)))
        return
    raise AssertionError("expected {}".format(exc))


pytest.raises = _raises
sys.modules["pytest"] = pytest

test_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, test_dir)

modules = sorted(os.path.splitext(os.path.basename(p))[0]
                 for p in glob.glob(os.path.join(test_dir, "test_*.py")))

fails = total = 0
for mod_name in modules:
    try:
        t = importlib.import_module(mod_name)
    except ImportError as e:
        print("SKIP", mod_name, "(import failed: {})".format(e))
        continue
    fns = [f for n, f in sorted(vars(t).items()) if n.startswith("test_") and callable(f)]
    for f in fns:
        if inspect.signature(f).parameters:
            print("SKIP", mod_name + "::" + f.__name__, "(requires pytest fixtures)")
            continue
        total += 1
        try:
            f()
            print("PASS", mod_name + "::" + f.__name__)
        except Exception:
            fails += 1
            print("FAIL", mod_name + "::" + f.__name__)
            traceback.print_exc()
print("{}/{} passed".format(total - fails, total))
sys.exit(1 if fails else 0)
