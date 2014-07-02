#!/usr/bin/env python

import os
import sys
from setuptools import setup
from setuptools.command.install import install
from distutils.command.build import build
from subprocess import call

# Setup.py for languages only


class build_me(build):
    def run(self):
        build.run(self)
        # if os.uname()[0] == 'Linux' and os.geteuid() == 0:
        #     call(['sudo', 'apt-get', 'install', 'build-essential'])
        #     call(['sudo', 'apt-get', 'install', 'g++-4.8'])
        #     call(['sudo', 'apt-get', 'install', 'cmake'])
        #     call(['sudo', 'apt-get', 'install', 'libboost-all-dev'])
        call(['mkdir', 'build'])
        os.chdir('build')
        call(['cmake', '..', '-DLANGUAGES=1'])
        call(['make'])
        os.chdir('..')


class install_me(install):
    def run(self):
        install.run(self)
        os.chdir('build')
        call(['make', 'install'])
        os.chdir('..')


class uninstall_me():
    def run(self):
        sys.stderr.write("This is currently a dummy method. "
                         "Nothing has been uninstalled.")


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='ethereum-serpent',
    version='1.4.11',
    description='Serpent compiler',
    maintainer='Vitalik Buterin',
    maintainer_email='v@buterin.com',
    license='WTFPL',
    url='http://www.ethereum.org/',
    long_description=read('README.md'),
    cmdclass={
        'build': build_me,
        'install': install_me,
        'uninstall': uninstall_me,
    }
)
