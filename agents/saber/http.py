"""HTTP interface setup for Saber Agent including registry and server configuration."""

from agent_kit.api.http import AgentRegistry, create_server
from agent_kit.api.mcp import run_mcp_stdio
from agent_kit.config import get_config

from .agent import SaberAgent
from .models import SaberRequest, SaberResponse


def create_saber_registry() -> AgentRegistry:
    """Create and configure agent registry for Saber Agent.

    Returns:
        Configured AgentRegistry
    """
    registry = AgentRegistry()

    registry.register(
        name="saber",
        agent_class=SaberAgent,
        description="Baseball statistics agent for MLB, minor league, and Negro League",
        request_model=SaberRequest,
        response_model=SaberResponse,
    )

    return registry


def create_saber_server():
    """Create HTTP server for Saber Agent with config from user directory.

    Returns:
        FastAPI application
    """
    config = get_config()
    registry = create_saber_registry()

    return create_server(registry, config.interfaces.http, config.interfaces.session_ttl)


def run_saber_stdio() -> None:
    """Run Saber Agent in MCP stdio mode for Claude Desktop integration."""
    registry = create_saber_registry()
    run_mcp_stdio(registry)
