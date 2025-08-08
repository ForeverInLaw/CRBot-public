#!/usr/bin/env python3
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
UTILS = os.path.join(ROOT, 'utils')
if UTILS not in sys.path:
    sys.path.insert(0, UTILS)

from utils.setup_adb import main

if __name__ == '__main__':
    main()
