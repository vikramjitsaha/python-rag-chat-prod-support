"""
Wrapper script to run data_loader from root directory
"""

import sys
from pathlib import Path

# Add source directory to path
current_dir = Path(__file__).parent
source_dir = current_dir / 'source'
sys.path.insert(0, str(source_dir))

if __name__ == "__main__":
    from data_loader import main
    main()
