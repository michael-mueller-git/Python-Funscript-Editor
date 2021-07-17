import setuptools
import os
import sys
import git # pip install gitpython

PACKAGE = 'funscript_editor'
DESCRIPTION = "A tool to create funscripts"
INCLUDE_REQUIREMENTS = False

TAGS = sorted(git.Repo('.').tags, key=lambda x: x.commit.committed_datetime) if os.path.exists('.git') else []
VERSION = str(TAGS[-1]) if len(TAGS) > 0 else "0.0.0"

src = [os.path.join('..', x) \
        for x in git.Git('.').ls_files().splitlines() \
        if x.startswith(PACKAGE+os.sep) \
        and os.path.exists(x)]

docs = [os.path.join('.', x) \
        for x in git.Git('.').ls_files().splitlines() \
        if x.startswith('docs'+os.sep) \
        and os.path.exists(x)]

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

if INCLUDE_REQUIREMENTS:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        requirements = [x.strip() for x in f.readlines()\
                if not any(a in x.lower() for a in ['opencv', 'pyqt'])]
                #NOTE: I recommend to install this packages with your favorite package manager.
                # Otherwise they will not work reliably. e.g. use pacman -Sy python-opencv python-pyqt5 ...
else:
    requirements = []

with open(os.path.join(PACKAGE, 'VERSION.txt'), 'w') as f:
    f.write(VERSION)

setuptools.setup(
    name=PACKAGE.replace('_', '-'),
    version=VERSION.replace('v', ''),
    author="btw i use arch",
    author_email="git@local",
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    entry_points={
        'console_scripts': [
            str(PACKAGE.replace('_', '-') + '=' + PACKAGE + '.__main__:main'),
        ]
    },
    install_requires=requirements,
    packages=[PACKAGE],
    package_data={PACKAGE: src},
    data_files=[(os.path.join(sys.prefix, 'share', PACKAGE, os.path.dirname(x)), [x]) for x in docs],
    python_requires=">=3.6",
    setup_requires=['wheel', 'gitpython'],
)
