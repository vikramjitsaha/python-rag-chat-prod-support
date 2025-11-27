"""
Wrapper script to run Streamlit app from root directory
"""

import sys
import os
from pathlib import Path

# Add source directory to path
current_dir = Path(__file__).parent
source_dir = current_dir / 'source'
sys.path.insert(0, str(source_dir))

# Run the app
if __name__ == "__main__":
    import streamlit.web.cli as stcli
    app_path = str(source_dir / "app.py")
    sys.argv = ["streamlit", "run", app_path]
    sys.exit(stcli.main())
