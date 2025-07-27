from setuptools import setup, find_packages

setup(
    name="cybersecurity-extraction-tools",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "transformers>=4.0.0", 
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "requests>=2.25.0",
        "beautifulsoup4>=4.9.0",
        "openpyxl>=3.0.7",
    ],
)