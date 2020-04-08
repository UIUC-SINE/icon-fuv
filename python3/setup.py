from setuptools import setup

setup(
    name='icon-fuv',
    version="1",
    packages=['iconfuv'],
    author="Ulas Kamaci",
    author_email="ukamaci2@illinois.edu",
    description="ICON FUV Nighttime Imaging Simulations",
    long_description=open('README.md').read(),
    license="GPLv3",
    keywords="icon fuv processing",
    url="https://github.com/uiuc-sine/icon-fuv",
    install_requires=[
        "matplotlib",
        "netCDF4",
        "numpy",
        "scipy",
        "opencv-python"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
