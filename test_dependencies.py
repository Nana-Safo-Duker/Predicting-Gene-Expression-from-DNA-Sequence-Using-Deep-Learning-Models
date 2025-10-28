#!/usr/bin/env python3
"""Test if all required dependencies are installed."""

import sys

print("Checking dependencies...")
print("=" * 50)

required_packages = [
    'numpy',
    'matplotlib',
    'seaborn',
    'scipy',
    'sklearn',
    'pandas'
]

missing = []

for package in required_packages:
    try:
        __import__(package)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package} - MISSING")
        missing.append(package)

print("=" * 50)

if missing:
    print(f"\nMissing packages: {', '.join(missing)}")
    print("\nTo install missing packages, run:")
    print("pip install -r requirements.txt")
    sys.exit(1)
else:
    print("\n✓ All dependencies are installed!")
    sys.exit(0)



