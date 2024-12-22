import pathlib
import setuptools

setuptools.setup(
    name="stresslog",
    version="1.2.1",
    description="Library for stress calculations from well logs",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    url="https://www.rocklab.in/stresslog",
    author="Arkadeep Ghosh",
    author_email="arkadeep_ghosh@rocklab.in",
    license="AGPL-3.0",
    #project_urls={
    #    "Documentation":"https://www.rocklab.in/stresslog_docs/"
    #    "Source":"https://github.com/GeoArkadeep/Stresslog"
    #},
    classifiers=[
        "Development Status :: 7-Release",
        "Intended Audience :: Engineers,Developers",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Utilities",
    ],
    python_requires=">=3.10,<3.12",
    install_requires=["pandas>=2.0","numpy","scipy","welly","pint","matplotlib"],
    packages=setuptools.find_packages(),
    include_package_data=True,    
)

