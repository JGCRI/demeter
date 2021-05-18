from setuptools import setup, find_packages


def readme():
    with open('README.md') as f:
        return f.read()


def get_requirements():
    with open('requirements.txt') as f:
        return f.read().split()


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
    install_requires=get_requirements(),
    dependency_links=["https://github.com/JGCRI/gcam_reader.git"],
    include_package_data=True
)