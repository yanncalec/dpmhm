#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=7.0',
'tensorflow', 'tensorflow-datasets', 'pandas', 'scipy', 'numpy', 'pydub', 'librosa', 'patool', 'tensorflow_addons']

test_requirements = [ ]

setup(
    author="Han Wang",
    author_email='han.wang@cea.fr',
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
    ],
    description="Diagnostic and Pronostic in Machine Health Monitoring",
    entry_points={
        'console_scripts': [
            'dpmhm=dpmhm.cli:main',
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='dpmhm',
    name='dpmhm',
    packages=find_packages(include=['dpmhm', 'dpmhm.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/yanncalec/dpmhm',
    version='0.1.0',
    zip_safe=False,
)
