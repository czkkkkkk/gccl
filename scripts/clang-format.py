#!/usr/bin/env python

import argparse
import os
import subprocess
import sys


default_dirs = [
    'src',
    'test',
]
ignored_dirs = [
]

clang_format = os.getenv('CLANG_FORMAT', 'clang-format')


def list_files(path=None):
    dirs = None
    if path is not None:
        if os.path.isdir(path):
            dirs = path
        else:
            if not os.path.exists(path):
                return None
            return [path]

    if dirs is None:
        dirs, files = default_dirs, []
        for d in dirs:
            cmd = 'find {} -name "*.h" -o -name "*.cc" -o -name "*.cu"'.format(
                d)
            res = subprocess.check_output(cmd, shell=True).rstrip()
            if(type(res) == bytes):
                res = res.decode('utf-8')
            files += res.split('\n')
        removes = []
        for f in files:
            for d in ignored_dirs:
                if f.find(d) != -1:
                    removes.append(f)
        for r in removes:
            files.remove(r)
        return files

    cmd = 'find {} -name "*.h" -o -name "*.cc" -o -name "*.cu"'.format(dirs)
    res = subprocess.check_output(cmd, shell=True).rstrip()
    if(type(res) == bytes):
        res = res.decode('utf-8')
    return res.split('\n')


def replace(files=None):
    if files is None:
        return

    for f in files:
        cmd = '{} -style=file -i {}'.format(clang_format, f)
        os.system(cmd)


def check(files=None):
    if files is None:
        return

    for f in files:
        cmd = '{} -style=file {} | diff -u {} -'.format(clang_format, f, f)
        result = os.system(cmd)
        if result != 0:
            print('Unexpected error')
            sys.exit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Clang-Format Tool')
    parser.add_argument('-o', choices=['replace', 'check'],
                        help='Replace for all c++ codes or check them only with clang-format')
    parser.add_argument('path', nargs='?',
                        help='Path to a specific file or a directory')
    args = parser.parse_args()

    files = None
    if args.path:
        files = list_files(args.path)
    else:
        files = list_files()

    if files is None:
        print('[Error] Path {} does not exist'.format(args.path))
        parser.print_help()

    if args.o is None:
        print('[Error] Need input argument for -o {}'.format(args.path))
        parser.print_help()

    if args.o == 'replace':
        replace(files)
    elif args.o == 'check':
        check(files)
