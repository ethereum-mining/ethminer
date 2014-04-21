#!/usr/bin/python
import os
import sys
import re
import json
import serpent


def main():
    if len(sys.argv) == 1:
        print "serpent <command> <arg1> <arg2> ..."
    else:
        cmd = sys.argv[2] if sys.argv[1][0] == '-' else sys.argv[1]
        if sys.argv[1] == '-s':
            args = re.findall(r'\S\S*', sys.stdin.read()) + sys.argv[3:]
        elif sys.argv[1] == '-B':
            args = [sys.stdin.read()] + sys.argv[3:]
        elif sys.argv[1] == '-b':
            args = [sys.stdin.read()[:-1]] + sys.argv[3:]  # remove trailing \n
        elif sys.argv[1] == '-j':
            args = [json.loads(sys.stdin.read())] + sys.argv[3:]
        elif sys.argv[1] == '-J':
            args = json.loads(sys.stdin.read()) + sys.argv[3:]
        elif sys.argv[1] == '-h':
            args = [x.decode('hex') for x in sys.argv[3:]]
        else:
            cmd = sys.argv[1]
            args = sys.argv[2:]
        try:
            args[0] = open(args[0]).read()
        except:
            pass
        o = getattr(serpent, cmd)(*args)
        if isinstance(o, (list, dict)):
            print json.dumps(o)
        else:
            print o.encode('hex')


if __name__ == '__main__':
    main()
