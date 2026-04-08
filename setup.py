"""
setup.py
Package configuration for the AI Email Intelligence Environment.
"""

from setuptools import setup, find_packages

setup(
    name="ai-email-intelligence-environment",
    version="1.0.0",
    description="OpenEnv-compliant AI environment for email management tasks",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "pydantic>=2.0.0,<3.0.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "inference": ["openai>=1.0.0"],
        "dev": ["pytest>=7.0.0", "pytest-cov>=4.0.0", "python-dotenv>=1.0.0"],
    },
)
