"""QuantScope: measure what a quantization mode costs, and keep the receipt.

The package is deliberately small and dependency-light: the core pipeline uses
numpy only, so a fresh clone can produce a first table on CPU without a network
call or a downloaded checkpoint.
"""

__version__ = "0.1.0"

from . import protocol, quant  # noqa: F401

__all__ = ["__version__", "protocol", "quant"]
