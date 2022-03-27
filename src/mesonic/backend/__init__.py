"""Package that contains all the Backend related modules.

The bases module contains the abstract classes that needs to be implemented by the
concrete Backend implementations.

The other modules contains the implementations of the abstract classes from the bases module.
"""
from typing import Type, Union

from mesonic.backend.bases import B, Backend


def start_backend(
    backend: Union[str, B, Type[B]] = "sc3nb", **backend_kwargs
) -> Backend:
    """Start a Backend.

    Parameters
    ----------
    backend : Union[str, Backend], optional
        Backend to be started, by default "sc3nb".
        Typically this takes the name of the backend as string.
        However it can also allows the user to provide a class
        or instance of the Backend type.

    Returns
    -------
    Backend
        The started Backend.

    Raises
    ------
    ValueError
        If the provided backend is not a subclass of the Backend class or
        if it is the abstract Backend class.
    NotImplementedError
        If an unknown backend is provided.

    """
    if isinstance(backend, type) and issubclass(backend, Backend):
        if backend is Backend:
            raise ValueError("Backend must be a subclass of {Backend} and not itself.")
        return backend(**backend_kwargs)
    elif isinstance(backend, Backend):
        if backend_kwargs:
            raise ValueError(
                "Cannot specify backend_kwargs when"
                " reusing already started backend instance"
            )
        return backend
    elif isinstance(backend, str):
        if backend == "sc3nb":
            from mesonic.backend.backend_sc3nb import BackendSC3NB

            return BackendSC3NB(**backend_kwargs)
    raise NotImplementedError(f"Unsupported backend: {backend}")
