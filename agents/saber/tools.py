"""Baseball statistics tools for Saber Agent."""

import asyncio
import logging
from typing import Any, cast

import statsapi
from pandas import DataFrame
from pybaseball import (
    batting_stats,
    pitching_stats,
    playerid_lookup,
    schedule_and_record,
    standings,
    statcast_batter,
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
    statcast_pitcher,
    statcast_pitcher_exitvelo_barrels,
    statcast_pitcher_expected_stats,
    statcast_pitcher_percentile_ranks,
    statcast_pitcher_spin_dir_comp,
)

logger = logging.getLogger(__name__)


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
        results["mlb_results"] = mlb_df.to_dict("records")
        results["status"] = "found"
        logger.info(f"Found {len(mlb_df)} MLB/Negro League matches for '{name}'")

    # Check MLB-StatsAPI (minor league players)
    minor_matches: list[dict[str, Any]] = cast(list[dict[str, Any]], await asyncio.to_thread(statsapi.lookup_player, name))
    if minor_matches:
        results["minor_results"] = minor_matches
        results["status"] = "found"
        logger.info(f"Found {len(minor_matches)} minor league matches for '{name}'")

    return results


# ========== Player Statistics (MLB/Negro League) ==========


async def get_player_batting_stats(
    player_name: str, start_season: int, end_season: int, league: str = "all"
) -> dict[str, Any]:
    """Get player batting statistics from pybaseball.

    Args:
        player_name: Player's full name
        start_season: Starting season (year)
        end_season: Ending season (year)
        league: "all", "nl", "al", or "mnl" (Negro League)

    Returns:
        Dictionary with batting statistics
    """
    # Fetch batting stats for the season range
    stats_df: DataFrame = await asyncio.to_thread(batting_stats, start_season, end_season, league=league)

    # Filter by player name
    player_stats = cast(DataFrame, stats_df[stats_df["Name"].str.contains(player_name, case=False, na=False)])

    if player_stats.empty:
        return {"error": f"No batting stats found for {player_name} ({start_season}-{end_season})"}

    # Convert to dictionary
    return {"stats": player_stats.to_dict("records"), "seasons": f"{start_season}-{end_season}"}


async def get_player_pitching_stats(
    player_name: str, start_season: int, end_season: int, league: str = "all"
) -> dict[str, Any]:
    """Get player pitching statistics from pybaseball.

    Args:
        player_name: Player's full name
        start_season: Starting season (year)
        end_season: Ending season (year)
        league: "all", "nl", "al", or "mnl" (Negro League)

    Returns:
        Dictionary with pitching statistics
    """
    # Fetch pitching stats for the season range
    stats_df: DataFrame = await asyncio.to_thread(pitching_stats, start_season, end_season, league=league)

    # Filter by player name
    player_stats = cast(DataFrame, stats_df[stats_df["Name"].str.contains(player_name, case=False, na=False)])

    if player_stats.empty:
        return {"error": f"No pitching stats found for {player_name} ({start_season}-{end_season})"}

    # Convert to dictionary
    return {"stats": player_stats.to_dict("records"), "seasons": f"{start_season}-{end_season}"}


async def get_player_career_stats(player_name: str, stat_type: str = "batting", league: str = "all") -> dict[str, Any]:
    """Get career aggregated stats across all available seasons.

    Args:
        player_name: Player's full name
        stat_type: "batting" or "pitching"
        league: "all", "nl", "al", or "mnl" (Negro League)

    Returns:
        Dictionary with career statistics
    """
    # Determine appropriate year range based on league
    if league == "mnl":
        start_year, end_year = 1920, 1948
    else:
        start_year, end_year = 1871, 2025

    if stat_type == "batting":
        return await get_player_batting_stats(player_name, start_year, end_year, league)
    elif stat_type == "pitching":
        return await get_player_pitching_stats(player_name, start_year, end_year, league)
    else:
        return {"error": f"Invalid stat_type: {stat_type}. Must be 'batting' or 'pitching'"}


# ========== Player Statistics (Minor League) ==========


async def get_minor_league_stats(player_id: int, level: str, season: int) -> dict[str, Any]:
    """Get minor league player statistics using MLB-StatsAPI.

    Args:
        player_id: MLB Advanced Media player ID
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
    stats: dict[str, Any] = cast(dict[str, Any], await asyncio.to_thread(statsapi.player_stat_data, player_id, group="[hitting,pitching]", type="season", sportId=sport_id, season=season))

    return {"player_id": player_id, "level": level, "season": season, "stats": stats}


async def get_player_progression(player_name: str) -> dict[str, Any]:
    """Track minor → major league career progression.

    Args:
        player_name: Player's full name

    Returns:
        Dictionary with progression data from minor to major leagues
    """
    # First lookup the player to get their ID
    lookup_result = await lookup_player(player_name)

    if lookup_result["status"] == "not_found":
        return {"error": f"Player '{player_name}' not found"}

    progression: dict[str, Any] = {"player_name": player_name, "mlb_stats": [], "minor_league_stats": []}

    # Get MLB career stats if available
    if lookup_result["mlb_results"]:
        mlb_career = await get_player_career_stats(player_name, "batting", "all")
        progression["mlb_stats"] = mlb_career

    # Get minor league stats if player ID available
    if lookup_result["minor_results"]:
        player_id = lookup_result["minor_results"][0].get("id")
        if player_id:
            # Try to get stats from each level
            for level in ["Rookie", "A", "High-A", "AA", "AAA"]:
                level_stats = await get_minor_league_stats(player_id, level, 2024)
                if "error" not in level_stats:
                    progression["minor_league_stats"].append(level_stats)

    return progression


# ========== Advanced Metrics (Statcast - 2015+) ==========


async def get_statcast_batter(player_id: int, start_date: str, end_date: str) -> dict[str, Any]:
    """Get Statcast batting metrics (exit velocity, launch angle, etc.).

    Args:
        player_id: MLB Advanced Media player ID
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format

    Returns:
        Dictionary with Statcast batting data
    """
    stats_df: DataFrame = await asyncio.to_thread(statcast_batter, start_date, end_date, player_id)

    if stats_df.empty:
        return {"error": f"No Statcast data found for player {player_id} ({start_date} to {end_date})"}

    # Calculate aggregates
    aggregates = {
        "total_pitches": len(stats_df),
        "avg_exit_velocity": stats_df["launch_speed"].mean() if "launch_speed" in stats_df else None,
        "avg_launch_angle": stats_df["launch_angle"].mean() if "launch_angle" in stats_df else None,
        "max_exit_velocity": stats_df["launch_speed"].max() if "launch_speed" in stats_df else None,
    }

    return {
        "player_id": player_id,
        "date_range": f"{start_date} to {end_date}",
        "aggregates": aggregates,
        "data": stats_df.to_dict("records")[:100],  # Limit to first 100 rows
    }


async def get_statcast_pitcher(player_id: int, start_date: str, end_date: str) -> dict[str, Any]:
    """Get Statcast pitching metrics (spin rate, velocity, etc.).

    Args:
        player_id: MLB Advanced Media player ID
        start_date: Start date in "YYYY-MM-DD" format
        end_date: End date in "YYYY-MM-DD" format

    Returns:
        Dictionary with Statcast pitching data
    """
    stats_df: DataFrame = await asyncio.to_thread(statcast_pitcher, start_date, end_date, player_id)

    if stats_df.empty:
        return {"error": f"No Statcast data found for pitcher {player_id} ({start_date} to {end_date})"}

    # Calculate aggregates
    aggregates = {
        "total_pitches": len(stats_df),
        "avg_velocity": stats_df["release_speed"].mean() if "release_speed" in stats_df else None,
        "avg_spin_rate": stats_df["release_spin_rate"].mean() if "release_spin_rate" in stats_df else None,
        "max_velocity": stats_df["release_speed"].max() if "release_speed" in stats_df else None,
    }

    return {
        "player_id": player_id,
        "date_range": f"{start_date} to {end_date}",
        "aggregates": aggregates,
        "data": stats_df.to_dict("records")[:100],  # Limit to first 100 rows
    }


async def get_statcast_batter_season(player_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast batting metrics for a player's season.

    Fetches complete Statcast profile by calling multiple pybaseball functions:
    - Exit velocity & barrels
    - Expected stats (xBA, xSLG, xwOBA)
    - Percentile rankings across all metrics
    - Performance against different pitch types

    Args:
        player_id: MLB Advanced Media player ID
        year: Season year (2015+)

    Returns:
        Dictionary with all Statcast batting metrics combined
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch exit velocity & barrels
    try:
        exitvelo_df: DataFrame = await asyncio.to_thread(statcast_batter_exitvelo_barrels, year, 25)
        if "player_id" in exitvelo_df.columns:
            player_exitvelo = cast(DataFrame, exitvelo_df[exitvelo_df["player_id"] == player_id])
            if not player_exitvelo.empty:
                result["exitvelo_barrels"] = player_exitvelo.to_dict("records")[0]
    except Exception:
        result["exitvelo_barrels"] = None

    # Fetch expected stats
    try:
        expected_df: DataFrame = await asyncio.to_thread(statcast_batter_expected_stats, year, 25)
        if "player_id" in expected_df.columns:
            player_expected = cast(DataFrame, expected_df[expected_df["player_id"] == player_id])
            if not player_expected.empty:
                result["expected_stats"] = player_expected.to_dict("records")[0]
    except Exception:
        result["expected_stats"] = None

    # Fetch percentile ranks
    try:
        percentile_df: DataFrame = await asyncio.to_thread(statcast_batter_percentile_ranks, year)
        if "player_id" in percentile_df.columns:
            player_percentile = cast(DataFrame, percentile_df[percentile_df["player_id"] == player_id])
            if not player_percentile.empty:
                result["percentile_ranks"] = player_percentile.to_dict("records")[0]
    except Exception:
        result["percentile_ranks"] = None

    # Fetch pitch arsenal performance
    try:
        arsenal_df: DataFrame = await asyncio.to_thread(statcast_batter_pitch_arsenal, year, 25)
        if "player_id" in arsenal_df.columns:
            player_arsenal = cast(DataFrame, arsenal_df[arsenal_df["player_id"] == player_id])
            if not player_arsenal.empty:
                result["pitch_arsenal"] = player_arsenal.to_dict("records")
    except Exception:
        result["pitch_arsenal"] = None

    # Check if we got any data
    has_data = any(v is not None for k, v in result.items() if k not in ["player_id", "year"])
    if not has_data:
        return {"error": f"No Statcast season data found for player {player_id} in {year}"}

    return result


async def get_statcast_pitcher_season(player_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast pitching metrics for a player's season.

    Fetches complete Statcast profile by calling multiple pybaseball functions:
    - Exit velocity & barrels allowed
    - Expected stats allowed (xBA, xSLG, xwOBA)
    - Percentile rankings across all metrics

    Note: Spin comparison requires specific pitch types, use get_pitcher_spin_comparison() instead.

    Args:
        player_id: MLB Advanced Media player ID
        year: Season year (2015+)

    Returns:
        Dictionary with all Statcast pitching metrics combined
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch exit velocity & barrels allowed
    try:
        exitvelo_df: DataFrame = await asyncio.to_thread(statcast_pitcher_exitvelo_barrels, year, 25)
        if "player_id" in exitvelo_df.columns:
            player_exitvelo = cast(DataFrame, exitvelo_df[exitvelo_df["player_id"] == player_id])
            if not player_exitvelo.empty:
                result["exitvelo_barrels"] = player_exitvelo.to_dict("records")[0]
    except Exception:
        result["exitvelo_barrels"] = None

    # Fetch expected stats allowed
    try:
        expected_df: DataFrame = await asyncio.to_thread(statcast_pitcher_expected_stats, year, 25)
        if "player_id" in expected_df.columns:
            player_expected = cast(DataFrame, expected_df[expected_df["player_id"] == player_id])
            if not player_expected.empty:
                result["expected_stats"] = player_expected.to_dict("records")[0]
    except Exception:
        result["expected_stats"] = None

    # Fetch percentile ranks
    try:
        percentile_df: DataFrame = await asyncio.to_thread(statcast_pitcher_percentile_ranks, year)
        if "player_id" in percentile_df.columns:
            player_percentile = cast(DataFrame, percentile_df[percentile_df["player_id"] == player_id])
            if not player_percentile.empty:
                result["percentile_ranks"] = player_percentile.to_dict("records")[0]
    except Exception:
        result["percentile_ranks"] = None

    # Check if we got any data
    has_data = any(v is not None for k, v in result.items() if k not in ["player_id", "year"])
    if not has_data:
        return {"error": f"No Statcast season data found for pitcher {player_id} in {year}"}

    return result


async def get_pitcher_spin_comparison(
    player_id: int, year: int, pitch_a: str = "FF", pitch_b: str = "CH", min_pitches: int = 100
) -> dict[str, Any]:
    """Compare spin direction/movement between two pitch types for a pitcher.

    Args:
        player_id: MLB Advanced Media player ID
        year: Season year (2015+)
        pitch_a: First pitch type to compare (default: "FF" = 4-seam fastball)
        pitch_b: Second pitch type to compare (default: "CH" = changeup)
        min_pitches: Minimum pitches thrown (default: 100)

    Common pitch types: FF (4-seam), SI (sinker), FC (cutter), SL (slider),
                        CU (curveball), CH (changeup), FS (splitter), KN (knuckle)

    Returns:
        Dictionary with spin comparison data
    """
    stats_df: DataFrame = await asyncio.to_thread(statcast_pitcher_spin_dir_comp, year, pitch_a, pitch_b, min_pitches)

    # Filter to specific player
    if "player_id" in stats_df.columns:
        player_stats = cast(DataFrame, stats_df[stats_df["player_id"] == player_id])
        if player_stats.empty:
            return {
                "player_id": player_id,
                "year": year,
                "pitch_a": pitch_a,
                "pitch_b": pitch_b,
                "error": f"No spin comparison data found for pitcher {player_id} in {year}",
            }
        return {"player_id": player_id, "year": year, "pitch_a": pitch_a, "pitch_b": pitch_b, "data": player_stats.to_dict("records")}

    # Return league-wide data if no player_id column
    return {"year": year, "pitch_a": pitch_a, "pitch_b": pitch_b, "data": stats_df.to_dict("records")}


async def get_fielding_stats(
    year: int, metric_type: str, position: int | None = None, min_attempts: str | int = "q"
) -> dict[str, Any]:
    """Get Statcast fielding metrics for a season.

    Args:
        year: Season year
        metric_type: Type of metric - "oaa", "outfield_directional", "catch_probability", "outfielder_jump", "catcher_poptime", "catcher_framing"
        position: Position number (3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF, 2=C) or None for all
        min_attempts: Minimum attempts or "q" for qualified

    Returns:
        Dictionary with fielding data
    """
    stats_df: DataFrame
    if metric_type == "oaa":
        stats_df = await asyncio.to_thread(statcast_outs_above_average, year, position or "all", min_attempts)
    elif metric_type == "outfield_directional":
        stats_df = await asyncio.to_thread(statcast_outfield_directional_oaa, year, min_attempts)
    elif metric_type == "catch_probability":
        stats_df = await asyncio.to_thread(statcast_outfield_catch_prob, year, min_attempts)
    elif metric_type == "outfielder_jump":
        stats_df = await asyncio.to_thread(statcast_outfielder_jump, year, min_attempts)
    elif metric_type == "catcher_poptime":
        stats_df = await asyncio.to_thread(statcast_catcher_poptime, year, min_2b_att=5, min_3b_att=0)
    elif metric_type == "catcher_framing":
        stats_df = await asyncio.to_thread(statcast_catcher_framing, year, min_attempts)
    else:
        return {"error": f"Invalid metric_type: {metric_type}"}

    return {"year": year, "metric_type": metric_type, "position": position, "data": stats_df.to_dict("records")}


async def get_fielding_season(player_id: int, year: int) -> dict[str, Any]:
    """Get ALL Statcast fielding metrics for a player's season.

    Fetches all available fielding metrics and filters to the specific player.
    Returns None for metrics where player has no data (e.g., outfield metrics for infielders).

    Args:
        player_id: MLB Advanced Media player ID
        year: Season year (2016+)

    Returns:
        Dictionary with all available fielding metrics
    """
    result: dict[str, Any] = {"player_id": player_id, "year": year}

    # Fetch outs above average (all positions)
    try:
        oaa_df: DataFrame = await asyncio.to_thread(statcast_outs_above_average, year, "all", "q")
        if "player_id" in oaa_df.columns:
            player_oaa = cast(DataFrame, oaa_df[oaa_df["player_id"] == player_id])
            if not player_oaa.empty:
                result["outs_above_average"] = player_oaa.to_dict("records")[0]
    except Exception:
        result["outs_above_average"] = None

    # Fetch outfield directional OAA
    try:
        directional_df: DataFrame = await asyncio.to_thread(statcast_outfield_directional_oaa, year, "q")
        if "player_id" in directional_df.columns:
            player_directional = cast(DataFrame, directional_df[directional_df["player_id"] == player_id])
            if not player_directional.empty:
                result["outfield_directional"] = player_directional.to_dict("records")[0]
    except Exception:
        result["outfield_directional"] = None

    # Fetch outfield catch probability
    try:
        catch_prob_df: DataFrame = await asyncio.to_thread(statcast_outfield_catch_prob, year, "q")
        if "player_id" in catch_prob_df.columns:
            player_catch_prob = cast(DataFrame, catch_prob_df[catch_prob_df["player_id"] == player_id])
            if not player_catch_prob.empty:
                result["catch_probability"] = player_catch_prob.to_dict("records")[0]
    except Exception:
        result["catch_probability"] = None

    # Fetch outfielder jump
    try:
        jump_df: DataFrame = await asyncio.to_thread(statcast_outfielder_jump, year, "q")
        if "player_id" in jump_df.columns:
            player_jump = cast(DataFrame, jump_df[jump_df["player_id"] == player_id])
            if not player_jump.empty:
                result["outfielder_jump"] = player_jump.to_dict("records")[0]
    except Exception:
        result["outfielder_jump"] = None

    # Fetch catcher poptime
    try:
        poptime_df: DataFrame = await asyncio.to_thread(statcast_catcher_poptime, year, min_2b_att=5, min_3b_att=0)
        if "player_id" in poptime_df.columns:
            player_poptime = cast(DataFrame, poptime_df[poptime_df["player_id"] == player_id])
            if not player_poptime.empty:
                result["catcher_poptime"] = player_poptime.to_dict("records")[0]
    except Exception:
        result["catcher_poptime"] = None

    # Fetch catcher framing (known to have parsing issues with some pybaseball versions)
    try:
        framing_df: DataFrame = await asyncio.to_thread(statcast_catcher_framing, year, "q")
        if "player_id" in framing_df.columns:
            player_framing = cast(DataFrame, framing_df[framing_df["player_id"] == player_id])
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
    standings_data: list[DataFrame] = cast(list[DataFrame], await asyncio.to_thread(standings, season))

    # standings() returns a list of 6 dataframes (one per division)
    all_standings: list[dict[str, Any]] = []
    for idx, div_df in enumerate(standings_data):
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
        team_stats = cast(DataFrame, stats_df[stats_df["Team"].str.contains(team_abbr, case=False, na=False)])
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
    boxscore: str = cast(str, await asyncio.to_thread(statsapi.boxscore, game_id))
    return {"game_id": game_id, "boxscore": boxscore}


# ========== Comparison & Analysis ==========


async def compare_players(player_names: list[str], seasons: str, stat_categories: list[str]) -> dict[str, Any]:
    """Side-by-side player comparison.

    Args:
        player_names: List of player names to compare
        seasons: Season range (e.g., "2023" or "2020-2023")
        stat_categories: List of stat categories to compare

    Returns:
        Dictionary with comparison data
    """
    # Parse season range
    if "-" in seasons:
        start, end = seasons.split("-")
        start_season, end_season = int(start), int(end)
    else:
        start_season = end_season = int(seasons)

    # Fetch stats for each player
    comparisons: list[dict[str, Any]] = []
    for player_name in player_names:
        batting = await get_player_batting_stats(player_name, start_season, end_season)
        pitching = await get_player_pitching_stats(player_name, start_season, end_season)

        comparisons.append({"player": player_name, "batting": batting, "pitching": pitching})

    return {"players": player_names, "seasons": seasons, "comparisons": comparisons}


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
        {"type": "web_search", "search_context_size": "high"},
        {
            "type": "function",
            "name": "lookup_player",
            "description": "Find and verify player name in MLB/Negro League (via pybaseball) and minor leagues (via MLB-StatsAPI). Returns matches from both sources. ALWAYS use this FIRST before querying any player statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Player name (full, partial, or last name only). Examples: 'Mike Trout', 'Trout', 'Satchel Paige'",
                    }
                },
                "required": ["name"],
            },
        },
        {
            "type": "function",
            "name": "get_player_batting_stats",
            "description": "Get batting statistics for a player across seasons (MLB or Negro League). Use after lookup_player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player's full name from lookup_player"},
                    "start_season": {"type": "integer", "description": "Starting season year"},
                    "end_season": {"type": "integer", "description": "Ending season year"},
                    "league": {
                        "type": "string",
                        "description": "League: 'all' (default), 'nl', 'al', or 'mnl' (Negro League)",
                        "enum": ["all", "nl", "al", "mnl"],
                    },
                },
                "required": ["player_name", "start_season", "end_season"],
            },
        },
        {
            "type": "function",
            "name": "get_player_pitching_stats",
            "description": "Get pitching statistics for a player across seasons (MLB or Negro League). Use after lookup_player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player's full name from lookup_player"},
                    "start_season": {"type": "integer", "description": "Starting season year"},
                    "end_season": {"type": "integer", "description": "Ending season year"},
                    "league": {
                        "type": "string",
                        "description": "League: 'all' (default), 'nl', 'al', or 'mnl' (Negro League)",
                        "enum": ["all", "nl", "al", "mnl"],
                    },
                },
                "required": ["player_name", "start_season", "end_season"],
            },
        },
        {
            "type": "function",
            "name": "get_player_career_stats",
            "description": "Get career aggregated stats across all available seasons for a player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_name": {"type": "string", "description": "Player's full name from lookup_player"},
                    "stat_type": {
                        "type": "string",
                        "description": "Type of stats: 'batting' or 'pitching'",
                        "enum": ["batting", "pitching"],
                    },
                    "league": {
                        "type": "string",
                        "description": "League: 'all' (default), 'nl', 'al', or 'mnl' (Negro League)",
                        "enum": ["all", "nl", "al", "mnl"],
                    },
                },
                "required": ["player_name"],
            },
        },
        {
            "type": "function",
            "name": "get_minor_league_stats",
            "description": "Get minor league player statistics using MLB-StatsAPI. Requires player_id from lookup_player.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "level": {
                        "type": "string",
                        "description": "Minor league level",
                        "enum": ["AAA", "AA", "High-A", "A", "Rookie"],
                    },
                    "season": {"type": "integer", "description": "Season year"},
                },
                "required": ["player_id", "level", "season"],
            },
        },
        {
            "type": "function",
            "name": "get_player_progression",
            "description": "Track minor to major league career progression for a player.",
            "parameters": {
                "type": "object",
                "properties": {"player_name": {"type": "string", "description": "Player's full name"}},
                "required": ["player_name"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_batter",
            "description": "Get Statcast batting metrics (exit velocity, launch angle, etc.). Only available 2015-present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["player_id", "start_date", "end_date"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_pitcher",
            "description": "Get Statcast pitching metrics (spin rate, velocity, etc.). Only available 2015-present.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "start_date": {"type": "string", "description": "Start date in YYYY-MM-DD format"},
                    "end_date": {"type": "string", "description": "End date in YYYY-MM-DD format"},
                },
                "required": ["player_id", "start_date", "end_date"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_batter_season",
            "description": "Get ALL Statcast batting metrics for a season (2015+). Returns complete profile: exit velocity, barrels, expected stats (xBA/xSLG/xwOBA), percentile ranks, and performance vs pitch types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "year": {"type": "integer", "description": "Season year (2015 or later)"},
                },
                "required": ["player_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_statcast_pitcher_season",
            "description": "Get ALL Statcast pitching metrics for a season (2015+). Returns complete profile: exit velocity allowed, barrels allowed, expected stats allowed (xBA/xSLG/xwOBA), and percentile ranks.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "year": {"type": "integer", "description": "Season year (2015 or later)"},
                },
                "required": ["player_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_pitcher_spin_comparison",
            "description": "Compare spin direction/movement between two pitch types for a pitcher (2015+). Common pitch types: FF (4-seam), SI (sinker), FC (cutter), SL (slider), CU (curveball), CH (changeup), FS (splitter), KN (knuckle).",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "year": {"type": "integer", "description": "Season year (2015 or later)"},
                    "pitch_a": {"type": "string", "description": "First pitch type (default: FF = 4-seam fastball)"},
                    "pitch_b": {"type": "string", "description": "Second pitch type (default: CH = changeup)"},
                    "min_pitches": {"type": "integer", "description": "Minimum pitches thrown (default: 100)"},
                },
                "required": ["player_id", "year"],
            },
        },
        {
            "type": "function",
            "name": "get_fielding_stats",
            "description": "Get Statcast fielding metrics (2016+). Includes OAA, outfield metrics, and catcher metrics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "year": {"type": "integer", "description": "Season year (2016 or later)"},
                    "metric_type": {
                        "type": "string",
                        "description": "Type of fielding metric",
                        "enum": [
                            "oaa",
                            "outfield_directional",
                            "catch_probability",
                            "outfielder_jump",
                            "catcher_poptime",
                            "catcher_framing",
                        ],
                    },
                    "position": {
                        "type": "integer",
                        "description": "Position number: 2=C, 3=1B, 4=2B, 5=3B, 6=SS, 7=LF, 8=CF, 9=RF (optional, None for all)",
                    },
                    "min_attempts": {
                        "type": "string",
                        "description": "Minimum attempts: 'q' for qualified or integer (default 'q')",
                    },
                },
                "required": ["year", "metric_type"],
            },
        },
        {
            "type": "function",
            "name": "get_fielding_season",
            "description": "Get ALL Statcast fielding metrics for a player's season (2016+). Returns complete fielding profile: outs above average, outfield directional OAA, catch probability, jump, catcher poptime, and framing. Metrics not applicable to player's position will be None.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_id": {"type": "integer", "description": "MLB Advanced Media player ID from lookup_player"},
                    "year": {"type": "integer", "description": "Season year (2016 or later)"},
                },
                "required": ["player_id", "year"],
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
            "name": "compare_players",
            "description": "Side-by-side comparison of multiple players' statistics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "player_names": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of player names to compare",
                    },
                    "seasons": {"type": "string", "description": "Season range (e.g., '2023' or '2020-2023')"},
                    "stat_categories": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of stat categories to compare",
                    },
                },
                "required": ["player_names", "seasons", "stat_categories"],
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
    ]


async def execute_tool(tool_name: str, tool_args: dict[str, Any]) -> dict[str, Any]:
    """Execute tool by name.

    Args:
        tool_name: Name of the tool to execute
        tool_args: Arguments for the tool

    Returns:
        Tool execution result
    """
    if tool_name == "lookup_player":
        return await lookup_player(**tool_args)
    elif tool_name == "get_player_batting_stats":
        return await get_player_batting_stats(**tool_args)
    elif tool_name == "get_player_pitching_stats":
        return await get_player_pitching_stats(**tool_args)
    elif tool_name == "get_player_career_stats":
        return await get_player_career_stats(**tool_args)
    elif tool_name == "get_minor_league_stats":
        return await get_minor_league_stats(**tool_args)
    elif tool_name == "get_player_progression":
        return await get_player_progression(**tool_args)
    elif tool_name == "get_statcast_batter":
        return await get_statcast_batter(**tool_args)
    elif tool_name == "get_statcast_pitcher":
        return await get_statcast_pitcher(**tool_args)
    elif tool_name == "get_statcast_batter_season":
        return await get_statcast_batter_season(**tool_args)
    elif tool_name == "get_statcast_pitcher_season":
        return await get_statcast_pitcher_season(**tool_args)
    elif tool_name == "get_pitcher_spin_comparison":
        return await get_pitcher_spin_comparison(**tool_args)
    elif tool_name == "get_fielding_stats":
        return await get_fielding_stats(**tool_args)
    elif tool_name == "get_fielding_season":
        return await get_fielding_season(**tool_args)
    elif tool_name == "get_team_standings":
        return await get_team_standings(**tool_args)
    elif tool_name == "get_team_stats":
        return await get_team_stats(**tool_args)
    elif tool_name == "get_team_schedule":
        return await get_team_schedule(**tool_args)
    elif tool_name == "get_game_boxscore":
        return await get_game_boxscore(**tool_args)
    elif tool_name == "compare_players":
        return await compare_players(**tool_args)
    elif tool_name == "get_league_leaders":
        return await get_league_leaders(**tool_args)
    else:
        raise ValueError(f"Unknown tool: {tool_name}")
