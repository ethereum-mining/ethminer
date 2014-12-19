#!/usr/bin/python
# cpp-ethereum build script
# to be used from CI server, or to build locally
# uses python instead of bash script for better cross-platform support

# TODO Initial version. Needs much more improvements

import argparse
import os
import subprocess

def build_dependencies():
   if os.path.exists("extdep"):
       os.chdir("extdep")
       if not os.path.exists("build"):
            os.makedirs("build") 
       os.chdir("build")
       subprocess.check_call(["cmake", ".."])
       subprocess.check_call("make")

parser = argparse.ArgumentParser()
parser.add_argument("cmd", help="what to build")

args = parser.parse_args()
if args.cmd == "dep":
    build_dependencies()
    
