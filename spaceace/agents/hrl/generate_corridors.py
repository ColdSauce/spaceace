"""Moved to spaceace.tools.generate_corridors. This shim keeps old imports working."""

from spaceace.tools.generate_corridors import *  # noqa: F401,F403
from spaceace.tools.generate_corridors import main

if __name__ == "__main__":
    main()
