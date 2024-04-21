#!/bin/env python3

import os
import platform
import argparse

from funscript_editor.api import show_editor, generate_funscript

def main():
    """ CLI Main Function """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", action = 'store_true', help = "Run only the generator")
    parser.add_argument("--multiaxis", action = 'store_true', help = "Show options for multiaxis output")
    parser.add_argument("--no-tracking", action = 'store_true', help = "Use previous tracking result")
    parser.add_argument("--logs", action = 'store_true', help = "Enable logging")
    parser.add_argument("--stdout", action = 'store_true', help = "Enable stdout logging")
    parser.add_argument("-i", "--input", type = str, help = "Video File")
    parser.add_argument("-o", "--output", type = str, default = "/tmp/funscript_actions.json", help = "Output Path")
    parser.add_argument("-s", "--start", type = float, default = 0.0, help = "Start Time in Milliseconds")
    parser.add_argument("-e", "--end", type = float, default = -1.0, help = "End/Stop Time in Milliseconds")
    args = parser.parse_args()

    if os.getcwd() not in os.environ['PATH']:
        os.environ['PATH'] = os.getcwd() + os.sep + os.environ['PATH']

    if platform.system().lower().startswith("linux") or os.path.abspath(__file__).startswith("/nix"):
        # pynput does not work well with native wayland so we use xwayland to get proper keyboard inputs
        if os.environ.get('DISPLAY'):
            print("Warning: Force QT_QPA_PLATFORM=xcb for better user experience")
            os.environ['QT_QPA_PLATFORM'] = "xcb"

    if not args.generator: show_editor()
    else: generate_funscript(args.input, args.start, args.end, args.output, args.multiaxis, args.no_tracking, args.logs, args.stdout)
