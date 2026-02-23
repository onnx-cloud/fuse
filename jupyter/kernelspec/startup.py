# Deprecated: startup stub kept for compatibility
# The canonical startup loader is now implemented at `jupyter/scripts/install_startup.py` and
# installed into the container during image build. This stub intentionally does nothing.

import warnings
warnings.warn("Deprecated startup: use jupyter/scripts/install_startup.py instead")
