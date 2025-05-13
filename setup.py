from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="genetic-algorithms-udec",
    version="1.0.0",
    author="Johan Sebastian Rojas Ramirez, Julian Lara Beltran",
    author_email="johansrojas@ucundinamarca.edu.co",
    description="Librería de Algoritmos Genéticos para Machine Learning - Universidad de Cundinamarca",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Zaxazgames1/genetic-algorithms-udec",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research", 
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.6",
    install_requires=[
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=20.8b1",
            "flake8>=3.8.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ga-solar-optimizer=examples.solar_panel_example:main",
        ],
    },
)