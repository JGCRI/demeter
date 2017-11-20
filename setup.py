import sys

try:
    from setuptools import setup, find_packages
except ImportError:
    sys.stdout.write('Missing Package:  setuptools not found.  Demeter requires this to install.  Please install setuptools and retry.')
    sys.exit(1)


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


setup(
    name='demeter',
    version='1.0.0',
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/demeter',
    license='BSD 2-Clause',
    author='Chris R. Vernon; Yannick le Page',
    author_email='chris.vernon@pnnl.gov; niquya@gmail.com',
    description='A land use land cover disaggregation and land change analytic model',
    long_description=readme(),
    install_requires=get_requirements()
)