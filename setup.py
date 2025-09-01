#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="paddle-format",
    version="2.2.0",
    author="Wei-JL",
    author_email="",
    description="一个功能全面、高度自动化的VOC和COCO数据集处理工具",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Wei-JL/paddle_format",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=[
        "opencv-python>=4.5.0",
        "tqdm>=4.60.0",
        "numpy>=1.19.0",
        "Pillow>=8.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
        ],
    },
    entry_points={
        "console_scripts": [
            "paddle-format=code.use_code.simple_process_example:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.md", "*.txt", "*.yml", "*.yaml"],
    },
    keywords=[
        "computer vision",
        "dataset processing",
        "VOC format",
        "COCO format",
        "data cleaning",
        "machine learning",
        "deep learning",
        "object detection",
        "image annotation",
        "paddle",
    ],
    project_urls={
        "Bug Reports": "https://github.com/Wei-JL/paddle_format/issues",
        "Source": "https://github.com/Wei-JL/paddle_format",
        "Documentation": "https://github.com/Wei-JL/paddle_format/blob/main/docs/使用指南.md",
    },
)