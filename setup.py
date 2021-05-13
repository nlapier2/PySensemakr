import os
import setuptools
import versioneer


HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'README.md'), 'r') as fid:
	LONG_DESCRIPTION = fid.read()

with open("requirements.txt", "r") as fh:
    requirements = [line.strip() for line in fh]

setuptools.setup(
    name="Pysensemakr",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    author="Nathan LaPierre",
    author_email="nathanl2012@gmail.com",
    description="Pysensemakr: Python implementation of the original R sensemakr package",
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url="https://github.com/nlapier2/PySensemakr",
    packages=setuptools.find_packages(),
    package_data={'PySensemakr': ['data/cmash*', 'data/db_info*', 'data/organism_files/*']},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Mathematics",
        "License :: OSI Approved :: MIT License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: POSIX :: Linux",
    ],
    python_requires='>=3.8',
    install_requires=requirements,
)
