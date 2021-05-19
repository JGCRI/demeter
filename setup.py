from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


setup(
    name='demeter',
    version='1.2.0',
    python_requires=">=3.6",
    packages=find_packages(),
    url='https://github.com/IMMM-SFA/demeter',
    license='BSD 2-Clause',
    author='Chris R. Vernon; Yannick le Page; Caleb Braun',
    author_email='chris.vernon@pnnl.gov',
    description='A land use land cover disaggregation and land change analytic model',
    long_description=readme(),
    long_description_content_type="text/markdown",
    install_requires=['configobj==5.0.6',
                      'numpy==1.20.3',
                      'pandas>=1.2.4',
                      'scipy==1.6.3',
                      'requests==2.25.1',
                      'gcam_reader==1.2.0'],
    dependency_links=['git+https://github.com/JGCRI/gcam_reader@master#egg=gcam_reader-1.2.0'],
    include_package_data=True
)