"""Example of setup installation for CLI Python modules"""

from setuptools import setup, find_packages

setup(
    name="patient_info",
    version="0.0.1",
    description="CLI test module",
    install_requires=["click"],
    entry_points="""
    [console_scripts]
    patient_info=cli_example.patient:pid
    """,
    author="JD",
    author_email="j@d.com",
    packages=find_packages(),
)
