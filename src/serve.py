"""Entry point for running the Fuse HTTP server locally."""

if __name__ == "__main__":
    import os
    import uvicorn

    host = os.environ.get("FUSE_SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("FUSE_SERVER_PORT", 8000))
    uvicorn.run("src.server.app:app", host=host, port=port)
