from setuptools import setup

setup(
    name="thesis",
    url="https://github.com/AbhiUppal/seniorthesis",
    version="0.0.1",
    author="Abhinuv Uppal",
    author_email="auppal22@students.claremontmckenna.edu",
    maintainer="Abhinuv Uppal",
    maintainer_email="auppal22@students.claremontmckenna.edu",
    keywords="data analysis statistics optimization network graph \
        research reproducibility classical machine learning AI ML science",
    license="LICENSE.md",
    description="Learning graph (network) structures on market data",
    long_description=open("README.md").read(),
    py_modules=["thesis"],
    install_requires=[
        "aiohttp",
        "aiolimiter",
        "backoff",
        "bidict",
        "black==21.7b0",
        "datetime",
        "dvc[gdrive,gs]",
        "flake8==3.9.2",
        "matplotlib",
        "nbclient",
        "nbdime",
        "nbconvert",
        "nbdev",
        "nbformat",
        "notebook",
        "numpy",
        "openpyxl",
        "pandas",
        "pandasql",
        "python-dotenv",
        "pytz",
        "us==2.0.2",
        "yfinance",
    ],
)
