"""Saber Agent - Baseball statistics expert."""

import logging
from pathlib import Path

from agent_kit.agents.base_agent import BaseAgent
from agent_kit.api.progress import ProgressHandler
from agent_kit.clients.openai_client import OpenAIClient
from agent_kit.config.config import get_config
from agent_kit.prompts.loader import PromptLoader

from agents.saber.teams import get_team_reference
from agents.saber.tools import execute_tool, get_tool_definitions

logger = logging.getLogger(__name__)


class SaberAgent(BaseAgent):
    """Baseball statistics agent for MLB, minor league, and Negro League."""

    def __init__(self, openai_client: OpenAIClient, progress_handler: ProgressHandler):
        """Initialize Saber Agent."""
        super().__init__(openai_client, progress_handler)

        # Register custom prompt directory
        prompts_root = Path(__file__).parent.parent
        self.prompt_loader = PromptLoader(search_paths=[prompts_root])

        logger.info("SaberAgent initialized")

    async def process(self, query: str) -> str:
        """Process baseball statistics query."""
        logger.info(f"Processing query: {query[:100]}")

        # Render orchestrator with teams reference and query injected
        prompts = self.render_prompt("saber", "orchestrator", teams_reference=get_team_reference(), query=query)

        # Get max iterations from config
        config = get_config()
        agent_config = config.agent_configs.get(self.agent_type, {})
        max_iterations = agent_config.get("max_iterations", 20)

        # Execute tool conversation
        response = await self.execute_tool_conversation(
            instructions=prompts["instructions"],
            initial_input=[{"role": "user", "content": prompts["user"]}],
            tools=get_tool_definitions(),
            tool_executor=execute_tool,
            max_iterations=max_iterations,
            previous_response_id=self.last_response_id,
            response_format=None,
        )

        # Extract text from response
        if response.output_text:
            logger.info("Successfully processed query")
            return response.output_text
        return response.output[0].text
