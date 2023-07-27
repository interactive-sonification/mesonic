import pytest

from mesonic.backend import start_backend
from mesonic.backend.backend_sc3nb import BackendSC3NB
from mesonic.backend.bases import Backend

__author__ = "Dennis Reinsch"
__copyright__ = "Dennis Reinsch"
__license__ = "MIT"


def test_start_backend():
    with pytest.raises(NotImplementedError):
        start_backend("xy")

    with pytest.raises(ValueError):
        start_backend(Backend)

    sc3nb_kwargs = dict(
        start_server=True, start_sclang=False, with_blip=False, console_logging=False
    )
    b1 = start_backend("sc3nb", **sc3nb_kwargs)
    assert isinstance(b1, BackendSC3NB)

    b2 = start_backend(BackendSC3NB, **sc3nb_kwargs)
    assert isinstance(b2, BackendSC3NB)
    with pytest.raises(ValueError):
        start_backend(b2, **sc3nb_kwargs)
    b2c = start_backend(b2)

    assert b2c is b2
