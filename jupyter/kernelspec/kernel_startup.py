#!/usr/bin/env python
"""
IPython kernel startup script for Fuse.
Automatically loads the Fuse IPython extension when the kernel starts.
"""

if __name__ == '__main__':
    import sys
    # When run via ipykernel_launcher -f {connection_file}, sys.argv will be set appropriately.
    # We just need to import and setup before the kernel starts.
    try:
        from IPython import get_ipython
        ip = get_ipython()
        if ip is not None:
            # Load the Fuse IPython extension
            ip.magic('load_ext src.jupyter.magics')
    except Exception as e:
        # Log but don't fail; the kernel can still run without the extension
        import warnings
        warnings.warn(f"Failed to load Fuse IPython extension: {e}")
    
    # Continue with normal ipykernel startup
    from ipykernel_launcher import main
    sys.exit(main())
