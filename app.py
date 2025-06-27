# app.py
"""
The main entry point for the Crypto Signal Hunter application.

This script initializes and runs the Streamlit dashboard. To start the app,
run `streamlit run app.py` from the command line in the project's root directory.
"""

from ui.dashboard import Dashboard

# The `if __name__ == "__main__"` block is a standard Python construct.
# It ensures that the code inside it only runs when the script is executed
# directly, not when it's imported as a module into another script.
if __name__ == "__main__":
    app = Dashboard()
    app.run()