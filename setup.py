from setuptools import setup

setup(
    name='e01loss',
    version='1.0.1',
    description='A Python library for solving the exact 0-1 loss linear classification problem',
    url='https://github.com/XiHegrt/E01Loss',
    author='Xi He',
    author_email='xihegrt@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='GNU GPL-3.0',
    packages=['e01loss','e01loss.test'],
    include_package_data=True,
    package_data={'e01loss': ['test/*.csv']},
    install_requires=['numpy','cvxopt'],
)
