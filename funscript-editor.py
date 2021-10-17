#!/bin/env python3

import traceback
from funscript_editor.__main__ import main

if __name__ == '__main__':
    try:
        main()
    except SystemExit as e:
        pass
    except:
        traceback.print_exc()
        input()
