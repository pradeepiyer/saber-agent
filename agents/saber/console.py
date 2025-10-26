"""Minimal console for Saber Agent."""

from typing import cast

from agent_kit.api.console.server import SlashCommands

from agents.saber.agent import SaberAgent


class SaberCommands(SlashCommands):
    """Minimal console - routes all input to agent.process()."""

    async def handle_input(self, user_input: str) -> bool:
        """Route input to framework commands or agent.

        Args:
            user_input: User's input string

        Returns:
            True if input was handled
        """
        # Try framework commands first (/help, /clear, /sessions, /quit)
        if await super().handle_input(user_input):
            return True

        # Unknown slash command = error
        if user_input.startswith("/"):
            cmd = user_input.split()[0] if user_input.split() else "/"
            self.console.print(f"[red]Unknown command: {cmd}[/red]")
            self.console.print("Type [cyan]/help[/cyan] to see available commands")
            return True

        # Everything else goes to SaberAgent
        if not self.session_id:
            self.console.print("[red]Session not initialized[/red]")
            return True

        session = await self.session_store.get_session(self.session_id)
        if not session:
            self.console.print("[red]Session not found[/red]")
            return True

        agent = cast(SaberAgent, await session.use_agent(SaberAgent))
        response = await agent.process(user_input, continue_conversation=True)

        self.console.print("\n[bold green]Saber Agent:[/bold green]")
        self.console.print(response, markup=False)
        self.console.print()

        return True
