"""Standalone entry point for saber agent.

Run with: python -m agents.saber
"""

import asyncio
import sys

import uvicorn
from agent_kit.api.console import run_console
from agent_kit.config import setup_configuration
from agent_kit.utils import set_app_name

from .console import SaberCommands
from .http import create_saber_server, run_saber_stdio


def main():
    """Entry point wrapper."""
    try:
        # Set custom user directory before any imports that use get_user_dir()
        set_app_name("saber-agent")

        # Load configuration to determine which interfaces are enabled
        config = asyncio.run(setup_configuration())

        # Start appropriate interface(s)
        if config.interfaces.http.enabled:
            app = create_saber_server()
            uvicorn.run(app, host=config.interfaces.http.host, port=config.interfaces.http.port, log_level="info")
        elif config.interfaces.console.enabled:
            asyncio.run(run_console(SaberCommands))
        elif config.interfaces.mcp_stdio.enabled:
            run_saber_stdio()
        else:
            print("Error: No interfaces enabled. Check your configuration.")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\nGoodbye!")
        sys.exit(0)


if __name__ == "__main__":
    main()
