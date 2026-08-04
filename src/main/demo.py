#!/usr/bin/env python
"""Run the Legal Assistant web backend from a named Python entrypoint."""

from __future__ import annotations

import os

import uvicorn


def main() -> None:
    host = os.getenv("BACKEND_HOST", "0.0.0.0")
    port = int(os.getenv("BACKEND_PORT", "8010"))
    workers = int(os.getenv("BACKEND_WORKERS", "1"))

    uvicorn.run(
        "Web.api:app",
        host=host,
        port=port,
        workers=workers,
    )


if __name__ == "__main__":
    main()
