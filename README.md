# Saber Agent

Baseball statistics AI agent for MLB, minor league, and Negro League baseball (1920-1948).

## Features

- **MLB Stats**: 1871-present (traditional), 2002-present (advanced metrics), 2015-present (Statcast)
- **Negro League Stats**: 1920-1948 (7 major leagues, 2,300+ players)
- **Minor League Stats**: 2000s-present (Triple-A, Double-A, High-A, Single-A, Rookie)
- **Multiple Interfaces**: Console, REST API, MCP

## Installation

```bash
# Install dependencies
uv sync

# Configure OpenAI API key
export OPENAI_API_KEY="sk-..."

# Initialize configuration
uv run agent-kit init
# Edit ~/.saber-agent/config.yaml

# Run agent
uv run python -m agents.saber
```

## Example Queries

- "What was Satchel Paige's ERA in the Negro Leagues?"
- "Compare Josh Gibson and Babe Ruth's career home runs"
- "How did Juan Soto perform in Triple-A before his MLB debut?"
- "Show me Mike Trout's Statcast metrics from 2023"
- "What are the current AL East standings?"

## Architecture

Built on [agent-kit](https://github.com/pradeepiyer/agent-kit) framework.

## License

MIT
