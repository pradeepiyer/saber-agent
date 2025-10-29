"""Baseball statistics tools for Saber Agent."""

import asyncio
import logging
import time
from typing import Any

import pandas as pd
import statsapi
from pandas import DataFrame
from pybaseball import (
    batting_stats,
    pitching_stats,
    playerid_lookup,
    schedule_and_record,
    standings,
    statcast_batter_exitvelo_barrels,
    statcast_batter_expected_stats,
    statcast_batter_percentile_ranks,
    statcast_batter_pitch_arsenal,
    statcast_catcher_framing,
    statcast_catcher_poptime,
    statcast_outfield_catch_prob,
    statcast_outfield_directional_oaa,
    statcast_outfielder_jump,
    statcast_outs_above_average,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
    statcast_pitcher_percentile_ranks,
)

logger = logging.getLogger(__name__)


# ========== Multi-Year Query Batching Workaround ==========


async def _fetch_stats_by_year(
    start_year: int,
    end_year: int,
    stat_type: str,
    league: str = "all",
    player_name: str | None = None,
    fangraphs_id: int | None = None,
) -> DataFrame:
    """Fetch stats year-by-year to avoid FanGraphs HTTP 500/524 errors.

    FanGraphs returns HTTP 500/524 when querying data across wide date ranges,
    especially for historical players. This function works around that by fetching
    one year at a time and combining results. Years are fetched in batches to avoid
    overwhelming FanGraphs' server.

    When fangraphs_id is provided, uses FanGraphs' players parameter for 10x speedup
    by fetching only the specific player's data instead of all players.

    Args:
        start_year: Starting season year
        end_year: Ending season year
        stat_type: "batting" or "pitching"
        league: "all", "nl", "al", or "mnl" (Negro League)
        player_name: Optional player name to filter results (fallback if no ID)
        fangraphs_id: FanGraphs player ID for optimization (10x speedup)

    Returns:
        Combined DataFrame with all years' data
    """
    fetch_fn = batting_stats if stat_type == "batting" else pitching_stats
    BATCH_SIZE = 5  # Fetch 5 years at a time to avoid rate limiting

    # Determine if we can use players parameter optimization
    # Negro League batters have invalid FanGraphs IDs, so we fall back for them
    use_players_param = (
        fangraphs_id is not None
        and fangraphs_id > 0  # Valid ID (not -1)
        and not (league == "mnl" and stat_type == "batting")  # Negro League batting fallback
    )

    async def fetch_year(year: int) -> tuple[int, DataFrame | None]:
        """Fetch stats for a single year."""
        try:
            if use_players_param:
                # OPTIMIZED: Fetch only specific player (10x faster)
                year_df: DataFrame = await asyncio.to_thread(
                    fetch_fn, year, year, league=league, players=str(fangraphs_id), qual=0
                )
            else:
                # FALLBACK: Fetch all players (name-based filtering below)
                year_df: DataFrame = await asyncio.to_thread(fetch_fn, year, year, league=league)

            if not year_df.empty:
                league_label = "Negro League" if league == "mnl" else league.upper()
                method = "player ID" if use_players_param else "name filter"
                logger.info(
                    f"Fetched {stat_type} stats for {league_label} {year} via {method}: {len(year_df)} rows"
                )
                return (year, year_df)
            return (year, None)
        except Exception as e:
            logger.warning(f"Failed to fetch {stat_type} stats for {league} {year}: {e}")
            return (year, None)

    # Fetch years in batches to avoid overwhelming FanGraphs
    years = list(range(start_year, end_year + 1))
    all_data: list[DataFrame] = []

    for i in range(0, len(years), BATCH_SIZE):
        batch = years[i : i + BATCH_SIZE]
        logger.info(f"Fetching batch of {len(batch)} years: {batch[0]}-{batch[-1]}")

        batch_results = await asyncio.gather(*[fetch_year(year) for year in batch], return_exceptions=True)

        # Filter out exceptions and None results
        for result in batch_results:
            if isinstance(result, Exception):
                logger.warning(f"Exception during fetch: {result}")
                continue
            if isinstance(result, tuple):
                _, df = result
                if df is not None:
                    all_data.append(df)

    if not all_data:
        return DataFrame()

    combined_df = pd.concat(all_data, ignore_index=True)

    # Filter by player name if needed (only when not using players parameter)
    if not use_players_param and player_name and "Name" in combined_df.columns:
        combined_df = combined_df[combined_df["Name"].str.contains(player_name, case=False, na=False)]

    return combined_df


# ========== Player Lookup & Resolution ==========


async def lookup_player(name: str) -> dict[str, Any]:
    """Look up player in MLB/Negro League (pybaseball) and minor leagues (MLB-StatsAPI).

    Args:
        name: Player name (can be partial: "Trout", "Mike Trout", etc.)

    Returns:
        dict with:
            - mlb_results: list of matches from pybaseball (MLB + Negro League)
            - minor_results: list of matches from MLB-StatsAPI
            - status: "found" | "not_found"
    """
    results: dict[str, Any] = {"mlb_results": [], "minor_results": [], "status": "not_found"}

    # Parse name (handle "First Last" or "Last" or "Last, First")
    parts = name.replace(",", " ").split()
    if len(parts) >= 2:
        last_name = parts[-1]
        first_name = " ".join(parts[:-1])
    else:
        last_name = parts[0]
        first_name = ""

    # Check pybaseball (MLB + Negro League)
    mlb_df: DataFrame = await asyncio.to_thread(playerid_lookup, last_name, first_name)
    if not mlb_df.empty:
        # Standardize ID field names for consistency
        mlb_records = mlb_df.to_dict("records")
        for record in mlb_records:
            # Rename key_mlbam to player_id
            if "key_mlbam" in record:
                record["player_id"] = int(record["key_mlbam"]) if pd.notna(record["key_mlbam"]) else -1
            # Rename key_fangraphs to fangraphs_id
            if "key_fangraphs" in record:
                record["fangraphs_id"] = int(record["key_fangraphs"]) if pd.notna(record["key_fangraphs"]) else -1
        results["mlb_results"] = mlb_records
        results["status"] = "found"
        logger.info(f"Found {len(mlb_df)} MLB/Negro League matches for '{name}'")

    # Check MLB-StatsAPI (minor league players)
    minor_matches: list[dict[str, Any]] = await asyncio.to_thread(statsapi.lookup_player, name)
    if minor_matches:
        # Standardize: MLB-StatsAPI returns 'id', rename to 'player_id' for consistency
        for record in minor_matches:
            if "id" in record:
                record["player_id"] = record["id"]
            # Minor league players don't have FanGraphs IDs, set to -1
            record["fangraphs_id"] = -1
        results["minor_results"] = minor_matches
        results["status"] = "found"
        logger.info(f"Found {len(minor_matches)} minor league matches for '{name}'")

    return results


# ========== Player Statistics (MLB/Negro League) ==========


async def get_player_batting_stats(
    player_name: str,
    player_id: int,
    fangraphs_id: int,
    start_season: int,
    end_season: int,
    league: str = "all",
) -> dict[str, Any]:
    """Get player batting statistics from pybaseball.

    Args:
        player_name: Player's full name
        player_id: MLB Advanced Media player ID from lookup_player()
        fangraphs_id: FanGraphs player ID from lookup_player() (enables 10x speedup)
        start_season: Starting season (year)
        end_season: Ending season (year)
        league: "all", "nl", "al", or "mnl" (Negro League)

    Returns:
        Dictionary with batting statistics
    """
    # Use fangraphs_id for optimization when valid (handles invalid IDs internally)
    stats_df = await _fetch_stats_by_year(
        start_season, end_season, "batting", league, player_name, fangraphs_id
    )

    if stats_df.empty:
        return {"error": f"No batting stats found for {player_name} ({start_season}-{end_season})"}

    # Convert to dictionary
    return {"stats": stats_df.to_dict("records"), "seasons": f"{start_season}-{end_season}"}


async def get_player_pitching_stats(
    player_name: str,
    player_id: int,
    fangraphs_id: int,
    start_season: int,
    end_season: int,
    league: str = "all",
) -> dict[str, Any]:
    """Get player pitching statistics from pybaseball.

    Args:
        player_name: Player's full name
        player_id: MLB Advanced Media player ID from lookup_player()
        fangraphs_id: FanGraphs player ID from lookup_player() (enables 10x speedup)
        start_season: Starting season (year)
        end_season: Ending season (year)
        league: "all", "nl", "al", or "mnl" (Negro League)

    Returns:
        Dictionary with pitching statistics
    """
    # Use fangraphs_id for optimization when valid (handles invalid IDs internally)
    stats_df = await _fetch_stats_by_year(
        start_season, end_season, "pitching", league, player_name, fangraphs_id
    )

    if stats_df.empty:
        return {"error": f"No pitching stats found for {player_name} ({start_season}-{end_season})"}

    # Convert to dictionary
    return {"stats": stats_df.to_dict("records"), "seasons": f"{start_season}-{end_season}"}


# ========== Player Statistics (Minor League) ==========


async def get_minor_league_stats(
    player_id: int, fangraphs_id: int, level: str, season: int
) -> dict[str, Any]:
    """Get minor league player statistics using MLB-StatsAPI.

    Args:
        player_id: MLB Advanced Media player ID (used)
        fangraphs_id: FanGraphs player ID from lookup_player() (not used by this tool)
        level: "AAA", "AA", "High-A", "A", or "Rookie"
        season: Season year

    Returns:
        Dictionary with minor league statistics
    """
    sport_ids = {"AAA": 11, "AA": 12, "High-A": 13, "A": 14, "Rookie": 16}

    sport_id = sport_ids.get(level)
    if not sport_id:
        return {"error": f"Invalid level: {level}. Must be one of {list(sport_ids.keys())}"}

    # Fetch player stats using statsapi
    stats: dict[str, Any] = await asyncio.to_thread(
        statsapi.player_stat_data,
        player_id,
        group="[hitting,pitching]",
        type="season",
        sportId=sport_id,
        season=season,
    )

    return {"player_id": player_id, "level": level, "season": season, "stats": stats}


async def get_player_progression(player_id: int, fangraphs_id: int) -> dict[str, Any]:
    """Track minor → major league career progression.

    Fetches player info to determine draft year and MLB debut, then searches
    for minor league stats across all levels during that timeframe.

    Args:
        player_id: MLB Advanced Media player ID (used)
        fangraphs_id: FanGraphs player ID from lookup_player() (not used by this tool)

    Returns:
        Dictionary with progression data including draft year, debut date, and all minor league stats
    """
    import datetime

    import requests

    # Get player info to find draft year and MLB debut
    url = f"https://statsapi.mlb.com/api/v1/people/{player_id}"
    response = await asyncio.to_thread(requests.get, url)
    person_data = response.json().get("people", [{}])[0]

    draft_year = person_data.get("draftYear")
    mlb_debut = person_data.get("mlbDebutDate")

    # Determine search range
    current_year = datetime.datetime.now().year
    start_year = draft_year if draft_year else current_year - 10
    end_year = int(mlb_debut[:4]) if mlb_debut else current_year

    progression: dict[str, Any] = {
        "player_id": player_id,
        "draft_year": draft_year,
        "mlb_debut": mlb_debut,
        "minor_league_stats": [],
    }

    # Search only the relevant years (draft → debut)
    for year in range(start_year, end_year + 1):
        for level in ["Rookie", "A", "High-A", "AA", "AAA"]:
            level_stats = await get_minor_league_stats(player_id, fangraphs_id, level, year)
            # Only add if we got actual stats (not just an error or empty result)
            if "error" not in level_stats and level_stats.get("stats"):
                progression["minor_league_stats"].append(level_stats)

    return progression


# ========== Advanced Metrics (Statcast - 2015+) ==========


async def get_statcast_batter_season(player_id: int, fangraphs_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast batting metrics for a player's season.

    Fetches complete Statcast profile by calling multiple pybaseball functions:
    - Exit velocity & barrels
    - Expected stats (xBA, xSLG, xwOBA)
    - Percentile rankings across all metrics
    - Performance against different pitch types

    Args:
        player_id: MLB Advanced Media player ID (used)
        fangraphs_id: FanGraphs player ID from lookup_player() (not used by this tool)
        year: Season year (2015+)

    Returns:
        Dictionary with all Statcast batting metrics combined
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch exit velocity & barrels
    try:
        exitvelo_df: DataFrame = await asyncio.to_thread(statcast_batter_exitvelo_barrels, year, 25)
        if "player_id" in exitvelo_df.columns:
            player_exitvelo = exitvelo_df[exitvelo_df["player_id"] == player_id]
            if not player_exitvelo.empty:
                result["exitvelo_barrels"] = player_exitvelo.to_dict("records")[0]
    except Exception:
        result["exitvelo_barrels"] = None

    # Fetch expected stats
    try:
        expected_df: DataFrame = await asyncio.to_thread(statcast_batter_expected_stats, year, 25)
        if "player_id" in expected_df.columns:
            player_expected = expected_df[expected_df["player_id"] == player_id]
            if not player_expected.empty:
                result["expected_stats"] = player_expected.to_dict("records")[0]
    except Exception:
        result["expected_stats"] = None

    # Fetch percentile ranks
    try:
        percentile_df: DataFrame = await asyncio.to_thread(statcast_batter_percentile_ranks, year)
        if "player_id" in percentile_df.columns:
            player_percentile = percentile_df[percentile_df["player_id"] == player_id]
            if not player_percentile.empty:
                result["percentile_ranks"] = player_percentile.to_dict("records")[0]
    except Exception:
        result["percentile_ranks"] = None

    # Fetch pitch arsenal performance
    try:
        arsenal_df: DataFrame = await asyncio.to_thread(statcast_batter_pitch_arsenal, year, 25)
        if "player_id" in arsenal_df.columns:
            player_arsenal = arsenal_df[arsenal_df["player_id"] == player_id]
            if not player_arsenal.empty:
                result["pitch_arsenal"] = player_arsenal.to_dict("records")
    except Exception:
        result["pitch_arsenal"] = None

    # Check if we got any data
    has_data = any(v is not None for k, v in result.items() if k not in ["player_id", "year"])
    if not has_data:
        return {"error": f"No Statcast season data found for player {player_id} in {year}"}

    return result


async def get_statcast_pitcher_season(player_id: int, fangraphs_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast pitching metrics for a player's season.

    Fetches complete Statcast profile by calling multiple pybaseball functions:
    - Exit velocity & barrels allowed
    - Expected stats allowed (xBA, xSLG, xwOBA)
    - Percentile rankings across all metrics

    Args:
        player_id: MLB Advanced Media player ID (used)
        fangraphs_id: FanGraphs player ID from lookup_player() (not used by this tool)
        year: Season year (2015+)

    Returns:
        Dictionary with all Statcast pitching metrics combined
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch exit velocity & barrels allowed
    try:
        exitvelo_df: DataFrame = await asyncio.to_thread(statcast_pitcher_exitvelo_barrels, year, 25)
        if "player_id" in exitvelo_df.columns:
            player_exitvelo = exitvelo_df[exitvelo_df["player_id"] == player_id]
            if not player_exitvelo.empty:
                result["exitvelo_barrels"] = player_exitvelo.to_dict("records")[0]
    except Exception:
        result["exitvelo_barrels"] = None

    # Fetch expected stats allowed
    try:
        expected_df: DataFrame = await asyncio.to_thread(statcast_pitcher_expected_stats, year, 25)
        if "player_id" in expected_df.columns:
            player_expected = expected_df[expected_df["player_id"] == player_id]
            if not player_expected.empty:
                result["expected_stats"] = player_expected.to_dict("records")[0]
    except Exception:
        result["expected_stats"] = None

    # Fetch percentile ranks
    try:
        percentile_df: DataFrame = await asyncio.to_thread(statcast_pitcher_percentile_ranks, year)
        if "player_id" in percentile_df.columns:
            player_percentile = percentile_df[percentile_df["player_id"] == player_id]
            if not player_percentile.empty:
                result["percentile_ranks"] = player_percentile.to_dict("records")[0]
    except Exception:
        result["percentile_ranks"] = None

    # Check if we got any data
    has_data = any(v is not None for k, v in result.items() if k not in ["player_id", "year"])
    if not has_data:
        return {"error": f"No Statcast season data found for pitcher {player_id} in {year}"}

    return result


async def get_fielding_season(player_id: int, fangraphs_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast fielding metrics for a player's season.

    Fetches all available fielding metrics and filters to the specific player.
    Returns None for metrics where player has no data (e.g., outfield metrics for infielders).

    Args:
        player_id: MLB Advanced Media player ID (used)
        fangraphs_id: FanGraphs player ID from lookup_player() (not used by this tool)
        year: Season year (2016+)

    Returns:
        Dictionary with all available fielding metrics
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch outs above average (all positions)
    try:
        oaa_df: DataFrame = await asyncio.to_thread(statcast_outs_above_average, year, "all", "q")
        if "player_id" in oaa_df.columns:
            player_oaa = oaa_df[oaa_df["player_id"] == player_id]
            if not player_oaa.empty:
                result["outs_above_average"] = player_oaa.to_dict("records")[0]
    except Exception:
        result["outs_above_average"] = None

    # Fetch outfield directional OAA
    try:
        directional_df: DataFrame = await asyncio.to_thread(statcast_outfield_directional_oaa, year, "q")
        if "player_id" in directional_df.columns:
            player_directional = directional_df[directional_df["player_id"] == player_id]
            if not player_directional.empty:
                result["outfield_directional"] = player_directional.to_dict("records")[0]
    except Exception:
        result["outfield_directional"] = None

    # Fetch outfield catch probability
    try:
        catch_prob_df: DataFrame = await asyncio.to_thread(statcast_outfield_catch_prob, year, "q")
        if "player_id" in catch_prob_df.columns:
            player_catch_prob = catch_prob_df[catch_prob_df["player_id"] == player_id]
            if not player_catch_prob.empty:
                result["catch_probability"] = player_catch_prob.to_dict("records")[0]
    except Exception:
        result["catch_probability"] = None

    # Fetch outfielder jump
    try:
        jump_df: DataFrame = await asyncio.to_thread(statcast_outfielder_jump, year, "q")
        if "player_id" in jump_df.columns:
            player_jump = jump_df[jump_df["player_id"] == player_id]
            if not player_jump.empty:
                result["outfielder_jump"] = player_jump.to_dict("records")[0]
    except Exception:
        result["outfielder_jump"] = None

    # Fetch catcher poptime
    try:
        poptime_df: DataFrame = await asyncio.to_thread(statcast_catcher_poptime, year, min_2b_att=5, min_3b_att=0)
        if "player_id" in poptime_df.columns:
            player_poptime = poptime_df[poptime_df["player_id"] == player_id]
            if not player_poptime.empty:
                result["catcher_poptime"] = player_poptime.to_dict("records")[0]
    except Exception:
        result["catcher_poptime"] = None

    # Fetch catcher framing (known to have parsing issues with some pybaseball versions)
    try:
        framing_df: DataFrame = await asyncio.to_thread(statcast_catcher_framing, year, "q")
        if "player_id" in framing_df.columns:
            player_framing = framing_df[framing_df["player_id"] == player_id]
            if not player_framing.empty:
                result["catcher_framing"] = player_framing.to_dict("records")[0]
    except Exception:
        result["catcher_framing"] = None

    return result


# ========== Team Statistics ==========


async def get_team_standings(season: int, division: str | None = None) -> dict[str, Any]:
    """Get division standings for season.

    Args:
        season: Season year
        division: Optional division filter (e.g., "AL East")

    Returns:
        Dictionary with standings data
    """
    standings_raw: Any = await asyncio.to_thread(standings, season)

    # Normalize to a list of DataFrames: pybaseball.standings may return either a single DataFrame
    # or a list of DataFrames depending on version; handle both safely.
    if isinstance(standings_raw, DataFrame):
        standings_list = [standings_raw]
    elif standings_raw is None:
        standings_list = []
    else:
        standings_list = list(standings_raw)

    # standings_list should now be an iterable of DataFrames (one per division)
    all_standings: list[dict[str, Any]] = []
    for idx, div_df in enumerate(standings_list):
        if not div_df.empty:
            div_name = div_df.columns[0] if len(div_df.columns) > 0 else f"Division_{idx}"
            all_standings.append({"division": div_name, "teams": div_df.to_dict("records")})

    # Filter by division if specified
    if division:
        all_standings = [s for s in all_standings if division.lower() in s["division"].lower()]

    return {"season": season, "standings": all_standings}


async def get_team_stats(team_abbr: str, season: int, stat_type: str = "batting") -> dict[str, Any]:
    """Get team aggregate statistics.

    Args:
        team_abbr: Team abbreviation (e.g., "NYY")
        season: Season year
        stat_type: "batting" or "pitching"

    Returns:
        Dictionary with team statistics
    """
    stats_df: DataFrame
    if stat_type == "batting":
        stats_df = await asyncio.to_thread(batting_stats, season, season)
    elif stat_type == "pitching":
        stats_df = await asyncio.to_thread(pitching_stats, season, season)
    else:
        return {"error": f"Invalid stat_type: {stat_type}"}

    # Filter by team if Team column exists
    if "Team" in stats_df.columns:
        team_stats = stats_df[stats_df["Team"].str.contains(team_abbr, case=False, na=False)]
        return {"team": team_abbr, "season": season, "type": stat_type, "stats": team_stats.to_dict("records")}
    else:
        return {"error": "Team information not available in stats data"}


async def get_team_schedule(team_abbr: str, season: int) -> dict[str, Any]:
    """Get full season schedule with results.

    Args:
        team_abbr: Team abbreviation (e.g., "NYY")
        season: Season year

    Returns:
        Dictionary with schedule and game results
    """
    schedule_df: DataFrame = await asyncio.to_thread(schedule_and_record, season, team_abbr)

    if schedule_df.empty:
        return {"error": f"No schedule found for {team_abbr} in {season}"}

    return {"team": team_abbr, "season": season, "games": schedule_df.to_dict("records")}


# ========== Game Data ==========


async def get_game_boxscore(game_id: str) -> dict[str, Any]:
    """Get detailed game box score.

    Args:
        game_id: MLB game ID

    Returns:
        Dictionary with box score data
    """
    boxscore: str = await asyncio.to_thread(statsapi.boxscore, game_id)
    return {"game_id": game_id, "boxscore": boxscore}


# ========== Comparison & Analysis ==========


async def get_league_leaders(season: int, stat_category: str, league: str = "MLB", limit: int = 10) -> dict[str, Any]:
    """Get top N performers in category.

    Args:
        season: Season year
        stat_category: Stat to rank by (e.g., "HR", "ERA")
        league: "MLB", "AL", "NL", or "mnl" (Negro League)
        limit: Number of top players to return

    Returns:
        Dictionary with league leaders
    """
    # Map league to pybaseball format
    league_map = {"MLB": "all", "AL": "al", "NL": "nl", "mnl": "mnl"}
    pybb_league = league_map.get(league, "all")

    # Try batting stats first
    batting_df: DataFrame = await asyncio.to_thread(batting_stats, season, season, league=pybb_league)
    if stat_category in batting_df.columns:
        leaders: DataFrame = batting_df.nlargest(limit, stat_category)
        return {
            "season": season,
            "stat": stat_category,
            "league": league,
            "type": "batting",
            "leaders": leaders.to_dict("records"),
        }

    # Try pitching stats
    pitching_df: DataFrame = await asyncio.to_thread(pitching_stats, season, season, league=pybb_league)
    if stat_category in pitching_df.columns:
        # For ERA and WHIP, we want smallest (use nsmallest)
        leaders: DataFrame
        if stat_category in ["ERA", "WHIP", "FIP"]:
            leaders = pitching_df.nsmallest(limit, stat_category)
        else:
            leaders = pitching_df.nlargest(limit, stat_category)

        return {
            "season": season,
            "stat": stat_category,
            "league": league,
            "type": "pitching",
            "leaders": leaders.to_dict("records"),
        }

    return {"error": f"Stat category '{stat_category}' not found"}


# ========== Tool Definitions & Execution ==========


def get_tool_definitions() -> list[dict[str, Any]]:
    """Return OpenAI tool definitions for all functions."""
    return [
        {
            "type": "function",
            "name": "lookup_player",
            "description": (
                "Find and verify player name in MLB/Negro League (via pybaseball) and minor leagues "
                "(via MLB-StatsAPI). Returns matches with player IDs AND career span (mlb_played_first, "
                "mlb_played_last years). USE these years as date ranges when querying player statistics "
                "to avoid timeouts. ALWAYS call this FIRST before querying any player statistics."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": (
                            "Player name (full, partial, or last name only). "
                            "Examples: 'Mike Trout', 'Trout', 'Satchel Paige'"
                        ),
                    }
                },
                "required": ["name"],
            },
        },
        {
            "type": "function",
            "name": "get_player_batting_stats",
            "description": (
                "Get batting statistics for a player across seasons (MLB or Negro League). "
                "REQUIRES player_id and fangraphs_id from lookup_player. Provides 10x speedup via FanGraphs player ID optimization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player's full name from lookup_player"},
                    "player_id": {
                        "type": "integer",
                        "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."
                    },
                    "fangraphs_id": {
                        "type": "integer",
                        "description": "FanGraphs player ID from lookup_player(). Enables 10x speedup. Always pass this."
                    },
                    "start_season": {"type": "integer", "description": "Starting season year"},
                    "end_season": {"type": "integer", "description": "Ending season year"},
                    "league": {
                        "type": "string",
                        "description": "League: 'all' (default), 'nl', 'al', or 'mnl' (Negro League)",
                        "enum": ["all", "nl", "al", "mnl"],
                    },
                },
                "required": ["player_name", "player_id", "fangraphs_id", "start_season", "end_season"],
            },
        },
        {
            "type": "function",
            "name": "get_player_pitching_stats",
            "description": (
                "Get pitching statistics for a player across seasons (MLB or Negro League). "
                "REQUIRES player_id and fangraphs_id from lookup_player. Provides 10x speedup via FanGraphs player ID optimization."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player's full name from lookup_player"},
                    "player_id": {
                        "type": "integer",
                        "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."
                    },
                    "fangraphs_id": {
                        "type": "integer",
                        "description": "FanGraphs player ID from lookup_player(). Enables 10x speedup. Always pass this."
                    },
                    "start_season": {"type": "integer", "description": "Starting season year"},
                    "end_season": {"type": "integer", "description": "Ending season year"},
                    "league": {
                        "type": "string",
                        "description": "League: 'all' (default), 'nl', 'al', or 'mnl' (Negro League)",
                        "enum": ["all", "nl", "al", "mnl"],
                    },
                },
                "required": ["player_name", "player_id", "fangraphs_id", "start_season", "end_season"],
            },
        },
        {
            "type": "function",
            "name": "get_minor_league_stats",
            "description": (
                "Get minor league player statistics using MLB-StatsAPI. REQUIRES both player_id and fangraphs_id from lookup_player."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."},
                    "fangraphs_id": {"type": "integer", "description": "FanGraphs player ID from lookup_player(). Always pass this."},
                    "level": {
                        "type": "string",
                        "description": "Minor league level",
                        "enum": ["AAA", "AA", "High-A", "A", "Rookie"],
                    },
                    "season": {"type": "integer", "description": "Season year"},
                },
                "required": ["player_id", "fangraphs_id", "level", "season"],
            },
        },
        {
            "type": "function",
            "name": "get_player_progression",
            "description": (
                "Track minor to major league career progression for a player. REQUIRES both player_id and fangraphs_id from lookup_player."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."},
                    "fangraphs_id": {"type": "integer", "description": "FanGraphs player ID from lookup_player(). Always pass this."}
                },
                "required": ["player_id", "fangraphs_id"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_batter_season",
            "description": (
                "Get ALL Statcast batting metrics for a season (2015+). Returns complete profile: "
                "exit velocity, barrels, expected stats (xBA/xSLG/xwOBA), percentile ranks, "
                "and performance vs pitch types. REQUIRES both player_id and fangraphs_id from lookup_player."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."},
                    "fangraphs_id": {"type": "integer", "description": "FanGraphs player ID from lookup_player(). Always pass this."},
                    "year": {"type": "integer", "description": "Season year (2015 or later)"},
                },
                "required": ["player_id", "fangraphs_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_pitcher_season",
            "description": (
                "Get ALL Statcast pitching metrics for a season (2015+). Returns complete profile: "
                "exit velocity allowed, barrels allowed, expected stats allowed (xBA/xSLG/xwOBA), "
                "and percentile ranks. REQUIRES both player_id and fangraphs_id from lookup_player."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."},
                    "fangraphs_id": {"type": "integer", "description": "FanGraphs player ID from lookup_player(). Always pass this."},
                    "year": {"type": "integer", "description": "Season year (2015 or later)"},
                },
                "required": ["player_id", "fangraphs_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_fielding_season",
            "description": (
                "Get ALL Statcast fielding metrics for a player's season (2016+). "
                "Returns complete fielding profile: outs above average, outfield directional OAA, "
                "catch probability, jump, catcher poptime, and framing. "
                "Metrics not applicable to player's position will be None. "
                "REQUIRES both player_id and fangraphs_id from lookup_player."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player(). Always pass this."},
                    "fangraphs_id": {"type": "integer", "description": "FanGraphs player ID from lookup_player(). Always pass this."},
                    "year": {"type": "integer", "description": "Season year (2016 or later)"},
                },
                "required": ["player_id", "fangraphs_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_team_standings",
            "description": "Get division standings for a season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "Season year"},
                    "division": {"type": "string", "description": "Optional division filter (e.g., 'AL East')"},
                },
                "required": ["season"],
            },
        },
        {
            "type": "function",
            "name": "get_team_stats",
            "description": "Get team aggregate statistics for a season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_abbr": {"type": "string", "description": "Team abbreviation (e.g., 'NYY', 'BOS')"},
                    "season": {"type": "integer", "description": "Season year"},
                    "stat_type": {
                        "type": "string",
                        "description": "Type of stats: 'batting' or 'pitching'",
                        "enum": ["batting", "pitching"],
                    },
                },
                "required": ["team_abbr", "season"],
            },
        },
        {
            "type": "function",
            "name": "get_team_schedule",
            "description": "Get full season schedule with game results for a team.",
            "parameters": {
                "type": "object",
                "properties": {
                    "team_abbr": {"type": "string", "description": "Team abbreviation (e.g., 'NYY', 'BOS')"},
                    "season": {"type": "integer", "description": "Season year"},
                },
                "required": ["team_abbr", "season"],
            },
        },
        {
            "type": "function",
            "name": "get_game_boxscore",
            "description": "Get detailed game box score using MLB game ID.",
            "parameters": {
                "type": "object",
                "properties": {"game_id": {"type": "string", "description": "MLB game ID"}},
                "required": ["game_id"],
            },
        },
        {
            "type": "function",
            "name": "get_league_leaders",
            "description": "Get top N performers in a stat category for a season.",
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {"type": "integer", "description": "Season year"},
                    "stat_category": {
                        "type": "string",
                        "description": "Stat to rank by (e.g., 'HR', 'ERA', 'AVG', 'SO')",
                    },
                    "league": {
                        "type": "string",
                        "description": "League: 'MLB', 'AL', 'NL', or 'mnl' (Negro League)",
                        "enum": ["MLB", "AL", "NL", "mnl"],
                    },
                    "limit": {"type": "integer", "description": "Number of top players to return (default 10)"},
                },
                "required": ["season", "stat_category"],
            },
        },
        {"type": "web_search", "search_context_size": "low"},
    ]


async def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Execute tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result
    """
    start_time = time.perf_counter()
    logger.info(f"[TIMING] Starting tool: {tool_name}")

    try:
        result = None
        if tool_name == "lookup_player":
            result = await lookup_player(**tool_args)
        elif tool_name == "get_player_batting_stats":
            result = await get_player_batting_stats(**tool_args)
        elif tool_name == "get_player_pitching_stats":
            result = await get_player_pitching_stats(**tool_args)
        elif tool_name == "get_minor_league_stats":
            result = await get_minor_league_stats(**tool_args)
        elif tool_name == "get_player_progression":
            result = await get_player_progression(**tool_args)
        elif tool_name == "get_statcast_batter_season":
            result = await get_statcast_batter_season(**tool_args)
        elif tool_name == "get_statcast_pitcher_season":
            result = await get_statcast_pitcher_season(**tool_args)
        elif tool_name == "get_fielding_season":
            result = await get_fielding_season(**tool_args)
        elif tool_name == "get_team_standings":
            result = await get_team_standings(**tool_args)
        elif tool_name == "get_team_stats":
            result = await get_team_stats(**tool_args)
        elif tool_name == "get_team_schedule":
            result = await get_team_schedule(**tool_args)
        elif tool_name == "get_game_boxscore":
            result = await get_game_boxscore(**tool_args)
        elif tool_name == "get_league_leaders":
            result = await get_league_leaders(**tool_args)
        else:
            raise ValueError(f"Unknown tool: {tool_name}")

        duration = time.perf_counter() - start_time
        logger.info(f"[TIMING] Completed tool: {tool_name} in {duration:.3f}s")
        return result
    except Exception as e:
        duration = time.perf_counter() - start_time
        logger.error(f"[TIMING] Failed tool: {tool_name} after {duration:.3f}s - {type(e).__name__}: {e}")
        raise
