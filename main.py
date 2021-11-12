#!/bin/env python3
import multiprocessing
from funscript_editor.__main__ import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
