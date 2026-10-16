import effgen
import smol_harness


def test_effgen_pinned_version():
    assert effgen.__version__ == "1.3.0"


def test_package_importable():
    assert smol_harness.__doc__
