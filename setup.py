import re
import os
import sys
from setuptools import setup, find_packages


def get_version():
    regex = r"^__version__ = '(.*)'$"
    with open('bbmd/__init__.py', 'r') as f:
        text = f.read()
    return re.findall(regex, text, re.MULTILINE)[0]


if sys.argv[-1] == 'publish_test':
    os.system('python setup.py sdist upload -r https://testpypi.python.org/pypi')
    os.system('python setup.py bdist_wheel upload -r https://testpypi.python.org/pypi')
    sys.exit()

if sys.argv[-1] == 'publish_production':
    os.system('python setup.py sdist upload')
    os.system('python setup.py bdist_wheel upload')
    sys.exit()


def readme():
    with open('README.rst') as f:
        return f.read()

setup(
    name='bbmd',
    version=get_version(),
    description='Bayesian benchmark dose (BBMD) modeling using PyStan',
    long_description=readme(),
    url='https://bitbucket.org/geniussk/bayesian-bmd-stats',
    author='Kan Shao, Andy Shapiro',
    author_email='kshao@indiana.edu',
    license='TBD',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy>=1.12.0',
        'scipy>=0.19.0',
        'pystan==2.14.0.0',
        'matplotlib>=1.4.3, <1.5',
        'python-docx>=0.8.6',
        'pandas>=0.19.2',
        'openpyxl>=2.4.5',
    ],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        'console_scripts': [
            'compile_stan_models = bbmd.compilation:compile_stan',
        ]
    },
)
