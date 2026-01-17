from setuptools import setup, find_packages

setup(
    name="mcqgenerator",
    version="0.0.1",
    author="Vivek Kumar",
    author_email="kumarvivekgupta18012000@gmail.com",
    packages=find_packages(),
    install_requires=[
        "langchain",
        "langchain-google-genai",
        "streamlit",
        "python-dotenv",
        "PyPDF2",
        "requests",
        "beautifulsoup4"
    ]
)