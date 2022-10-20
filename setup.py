from setuptools import setup, find_packages


def readme():
    """Return the contents of the project README file."""
    with open('README.md') as f:
        return f.read()


setup(
    name='demeter',
    version='1.3.1',
    python_requires=">=3.8",
    packages=find_packages(),
    url='https://github.com/JGCRI/demeter',
    license='BSD 2-Clause',
    author='Chris R. Vernon',
    author_email='chris.vernon@pnnl.gov',
    description='A land use land cover change disaggregation model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=['configobj~=5.0.6',
                      'numpy~=1.20.3',
                      'pandas~=1.2.4',
                      'scipy~=1.6.3',
                      'requests~=2.20.0',
                      'gcamreader>=1.2.5',
                      'xarray >= 0.20.2'],
    include_package_data=True
)
