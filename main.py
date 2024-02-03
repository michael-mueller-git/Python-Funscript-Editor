#!/bin/env python3
import multiprocessing
import os
import sys

if not os.path.exists("funscript_editor"):
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))  

from funscript_editor.__main__ import main

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
