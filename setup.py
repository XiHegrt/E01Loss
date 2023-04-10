from setuptools import setup

setup(
    name='e01loss',
    version='0.0.8',
    description='A Python library for solving the exact 0-1 loss linear classification problem',
    url='https://github.com/XiHegrt/Exact-ML-Algorithms',
    author='Xi He',
    author_email='xihegrt@gmail.com',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    license='MIT License',
    packages=['e01loss','e01loss.test'],
    include_package_data=True,
    package_data={'e01loss': ['test/*.csv']},
    install_requires=['numpy','cvxopt'],
)