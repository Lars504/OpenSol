"""Entry point of ``python -m open_sol.gui [model.json]``."""
import sys

from .app import main

if __name__ == '__main__':
    sys.exit(main())
