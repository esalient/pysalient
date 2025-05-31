"""
Pytest configuration for pysalient tests.
"""
import sys
from pathlib import Path

# Add the project root to Python path so tests can import pysalient without installation
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
