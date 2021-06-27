#!/bin/env python3

import os
import argparse

from funscript_editor.api import show_editor, generate_funscript

def main():
    """ CLI Main Function """
    parser = argparse.ArgumentParser()
    parser.add_argument("--generator", action = 'store_true', help = "Run only the generator")
    parser.add_argument("-i", "--input", type = str, help = "Video File")
    parser.add_argument("-o", "--output", type = str, default = "/tmp/funscript_actions.csv", help = "Output Path")
    parser.add_argument("-s", "--start", type = int, default = 0, help = "Start Time in Milliseconds")
    args = parser.parse_args()

    if os.getcwd() not in os.environ['PATH']:
        os.environ['PATH'] = os.getcwd() + os.sep + os.environ['PATH']

    if not args.generator: show_editor()
    else: generate_funscript(args.input, args.start, args.output)
