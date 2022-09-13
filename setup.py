import os
from setuptools import setup

here = os.path.abspath(os.path.dirname(__file__))
with open("README.rst", 'r', encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='hifiscan',
    version='1.0.2',
    description='Optimize the audio quality of loudspeakers',
    long_description=long_description,
    packages=['hifiscan'],
    url='https://github.com/erdewit/hifiscan',
    author='Ewald R. de Wit',
    author_email='ewald.de.wit@gmail.com',
    license='BSD',
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: End Users/Desktop',
        'Topic :: Multimedia :: Sound/Audio :: Analysis',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3 :: Only',
    ],
    keywords='frequency impulse response audio spectrum equalizer',
    entry_points={
        'gui_scripts': ['hifiscan=hifiscan.app:main']
    },
    python_requires=">=3.8",
    install_requires=['eventkit', 'numba', 'numpy', 'PySide6', 'pyqtgraph',
                      'sounddevice']
)
