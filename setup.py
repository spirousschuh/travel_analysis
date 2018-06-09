import os

from setuptools import find_packages
from setuptools import setup

version = '0.1.2.dev0'

here = os.path.abspath(os.path.dirname(__file__))
README = open(os.path.join(here, 'README.md')).read()
CHANGES = open(os.path.join(here, 'CHANGES.md')).read()

requires = [
    'coverage', 'seaborn', 'pytest', 'pytest-cov',
    ]


setup(
    name='travel analysis',
    version=version,
    description="Forecast and optimize travel prices",
    long_description=README + '\n\n' + CHANGES,
    classifiers=[
        'Environment :: Console', 'Intended Audience :: Developers',
        'Programming Language :: Python :: 3.6'
    ],
    keywords='forecast optimize travel prices',
    author='spirousschuh    ',
    author_email='hasensilvester@gmail.com',
    url='https://github.com/spirousschuh/travel_analysis',
    license='None',
    packages=find_packages(),
    include_package_data=True,
    zip_safe=True,
    install_requires=requires,
    test_suite='test',
    entry_points={
        'console_scripts': [
        ]
    })
