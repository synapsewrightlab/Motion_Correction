from setuptools import find_packages, setup

def readme():
    with open("README.md") as f:
        return f.read()
    

setup(
    name="Motion_Correction",
    version="0.0.1",
    description="Code for motion correction",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/synapsewrightlab/Motion_Correction",
    author="William (Jake) Wright",
    license="MIT",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "Motion_Correct = Motion_Correction.gui.guiMC:main"
        ]
    }
)