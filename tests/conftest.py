import os


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: full model runs (deselect with '-m \"not slow\"')")

    # pin numerical library threads; oversubscription with the model's thread pools slows runs ~2.6x
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(var, "1")
