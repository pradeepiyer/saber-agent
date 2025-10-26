"""MLB and Negro League teams dictionary and reference generation."""

# Current MLB Teams (30)
MLB_TEAMS = {
    # American League East
    "BAL": {
        "full": "Baltimore Orioles",
        "city": "Baltimore",
        "aliases": ["Orioles", "O's"],
        "division": "AL East",
    },
    "BOS": {
        "full": "Boston Red Sox",
        "city": "Boston",
        "aliases": ["Red Sox", "Sox"],
        "division": "AL East",
    },
    "NYY": {
        "full": "New York Yankees",
        "city": "New York",
        "aliases": ["Yankees", "Yanks"],
        "division": "AL East",
    },
    "TB": {
        "full": "Tampa Bay Rays",
        "city": "Tampa Bay",
        "aliases": ["Rays"],
        "division": "AL East",
    },
    "TOR": {
        "full": "Toronto Blue Jays",
        "city": "Toronto",
        "aliases": ["Blue Jays", "Jays"],
        "division": "AL East",
    },
    # American League Central
    "CLE": {
        "full": "Cleveland Guardians",
        "city": "Cleveland",
        "aliases": ["Guardians"],
        "division": "AL Central",
    },
    "CWS": {
        "full": "Chicago White Sox",
        "city": "Chicago",
        "aliases": ["White Sox"],
        "division": "AL Central",
    },
    "DET": {
        "full": "Detroit Tigers",
        "city": "Detroit",
        "aliases": ["Tigers"],
        "division": "AL Central",
    },
    "KC": {
        "full": "Kansas City Royals",
        "city": "Kansas City",
        "aliases": ["Royals"],
        "division": "AL Central",
    },
    "MIN": {
        "full": "Minnesota Twins",
        "city": "Minnesota",
        "aliases": ["Twins"],
        "division": "AL Central",
    },
    # American League West
    "HOU": {
        "full": "Houston Astros",
        "city": "Houston",
        "aliases": ["Astros"],
        "division": "AL West",
    },
    "LAA": {
        "full": "Los Angeles Angels",
        "city": "Los Angeles",
        "aliases": ["Angels"],
        "division": "AL West",
    },
    "OAK": {
        "full": "Oakland Athletics",
        "city": "Oakland",
        "aliases": ["Athletics", "A's"],
        "division": "AL West",
    },
    "SEA": {
        "full": "Seattle Mariners",
        "city": "Seattle",
        "aliases": ["Mariners"],
        "division": "AL West",
    },
    "TEX": {
        "full": "Texas Rangers",
        "city": "Texas",
        "aliases": ["Rangers"],
        "division": "AL West",
    },
    # National League East
    "ATL": {
        "full": "Atlanta Braves",
        "city": "Atlanta",
        "aliases": ["Braves"],
        "division": "NL East",
    },
    "MIA": {
        "full": "Miami Marlins",
        "city": "Miami",
        "aliases": ["Marlins"],
        "division": "NL East",
    },
    "NYM": {
        "full": "New York Mets",
        "city": "New York",
        "aliases": ["Mets"],
        "division": "NL East",
    },
    "PHI": {
        "full": "Philadelphia Phillies",
        "city": "Philadelphia",
        "aliases": ["Phillies"],
        "division": "NL East",
    },
    "WSH": {
        "full": "Washington Nationals",
        "city": "Washington",
        "aliases": ["Nationals", "Nats"],
        "division": "NL East",
    },
    # National League Central
    "CHC": {
        "full": "Chicago Cubs",
        "city": "Chicago",
        "aliases": ["Cubs"],
        "division": "NL Central",
    },
    "CIN": {
        "full": "Cincinnati Reds",
        "city": "Cincinnati",
        "aliases": ["Reds"],
        "division": "NL Central",
    },
    "MIL": {
        "full": "Milwaukee Brewers",
        "city": "Milwaukee",
        "aliases": ["Brewers"],
        "division": "NL Central",
    },
    "PIT": {
        "full": "Pittsburgh Pirates",
        "city": "Pittsburgh",
        "aliases": ["Pirates"],
        "division": "NL Central",
    },
    "STL": {
        "full": "St. Louis Cardinals",
        "city": "St. Louis",
        "aliases": ["Cardinals"],
        "division": "NL Central",
    },
    # National League West
    "ARI": {
        "full": "Arizona Diamondbacks",
        "city": "Arizona",
        "aliases": ["Diamondbacks", "D-backs"],
        "division": "NL West",
    },
    "COL": {
        "full": "Colorado Rockies",
        "city": "Colorado",
        "aliases": ["Rockies"],
        "division": "NL West",
    },
    "LAD": {
        "full": "Los Angeles Dodgers",
        "city": "Los Angeles",
        "aliases": ["Dodgers"],
        "division": "NL West",
    },
    "SD": {
        "full": "San Diego Padres",
        "city": "San Diego",
        "aliases": ["Padres"],
        "division": "NL West",
    },
    "SF": {
        "full": "San Francisco Giants",
        "city": "San Francisco",
        "aliases": ["Giants"],
        "division": "NL West",
    },
}

# Negro League Teams (~40 prominent teams from 7 major leagues, 1920-1948)
NEGRO_LEAGUE_TEAMS = {
    # Negro National League (1920-1931, 1933-1948)
    "KC-MON": {
        "full": "Kansas City Monarchs",
        "city": "Kansas City",
        "league": "NNL/NAL",
        "years": "1920-1965",
    },
    "HG": {
        "full": "Homestead Grays",
        "city": "Homestead",
        "league": "NNL",
        "years": "1912-1950",
    },
    "CAG": {
        "full": "Chicago American Giants",
        "city": "Chicago",
        "league": "NNL",
        "years": "1920-1956",
    },
    "PC": {
        "full": "Pittsburgh Crawfords",
        "city": "Pittsburgh",
        "league": "NNL",
        "years": "1932-1938",
    },
    "STL-STARS": {
        "full": "St. Louis Stars",
        "city": "St. Louis",
        "league": "NNL",
        "years": "1920-1931",
    },
    "DS": {
        "full": "Detroit Stars",
        "city": "Detroit",
        "league": "NNL",
        "years": "1920-1931",
    },
    "IND-ABC": {
        "full": "Indianapolis ABCs",
        "city": "Indianapolis",
        "league": "NNL",
        "years": "1920-1926",
    },
    "CG": {
        "full": "Chicago Giants",
        "city": "Chicago",
        "league": "NNL",
        "years": "1920-1921",
    },
    "DM": {
        "full": "Dayton Marcos",
        "city": "Dayton",
        "league": "NNL",
        "years": "1920",
    },
    "NE": {
        "full": "Newark Eagles",
        "city": "Newark",
        "league": "NNL",
        "years": "1936-1948",
    },
    "PS": {
        "full": "Philadelphia Stars",
        "city": "Philadelphia",
        "league": "NNL",
        "years": "1934-1948",
    },
    "NYC": {
        "full": "New York Cubans",
        "city": "New York",
        "league": "NNL",
        "years": "1935-1950",
    },
    "BE": {
        "full": "Baltimore Elite Giants",
        "city": "Baltimore",
        "league": "NNL",
        "years": "1938-1951",
    },
    # Negro American League (1937-1948+)
    "BBB": {
        "full": "Birmingham Black Barons",
        "city": "Birmingham",
        "league": "NAL",
        "years": "1920-1960",
    },
    "MRS": {
        "full": "Memphis Red Sox",
        "city": "Memphis",
        "league": "NAL",
        "years": "1937-1962",
    },
    "CB": {
        "full": "Cleveland Buckeyes",
        "city": "Cleveland",
        "league": "NAL",
        "years": "1942-1950",
    },
    "IC": {
        "full": "Indianapolis Clowns",
        "city": "Indianapolis",
        "league": "NAL",
        "years": "1943-1960s",
    },
    # Eastern Colored League (1923-1928)
    "BBS": {
        "full": "Baltimore Black Sox",
        "city": "Baltimore",
        "league": "ECL/ANL",
        "years": "1923-1934",
    },
    "HC": {
        "full": "Hilldale Club",
        "city": "Philadelphia",
        "league": "ECL",
        "years": "1923-1932",
    },
    "BRG": {
        "full": "Brooklyn Royal Giants",
        "city": "Brooklyn",
        "league": "ECL",
        "years": "1905-1942",
    },
    "BG": {
        "full": "Bacharach Giants",
        "city": "Atlantic City",
        "league": "ECL",
        "years": "1916-1929",
    },
    "LG": {
        "full": "Lincoln Giants",
        "city": "New York",
        "league": "ECL",
        "years": "1911-1930",
    },
    "CS-E": {
        "full": "Cuban Stars (East)",
        "city": "New York",
        "league": "ECL",
        "years": "1923-1928",
    },
    # Other notable teams
    "CS-W": {
        "full": "Cuban Stars (West)",
        "city": "Cincinnati",
        "league": "NNL",
        "years": "1920-1930",
    },
    "CXG": {
        "full": "Cuban X-Giants",
        "city": "New York",
        "league": "Independent",
        "years": "1896-1906",
    },
    "NYB": {
        "full": "New York Black Yankees",
        "city": "New York",
        "league": "NNL",
        "years": "1931-1948",
    },
}


def get_team_reference() -> str:
    """Generate comprehensive team reference for prompt injection.

    Returns:
        Formatted markdown string with all MLB and Negro League teams
    """
    lines = ["# MLB Teams (Current)\n"]

    for abbr, info in sorted(MLB_TEAMS.items()):
        aliases = ", ".join(info["aliases"])
        lines.append(f"- **{abbr}**: {info['full']} ({info['division']}) - also: {aliases}")

    lines.append("\n# Negro League Teams (1920-1948)\n")

    for abbr, info in sorted(NEGRO_LEAGUE_TEAMS.items()):
        lines.append(f"- **{abbr}**: {info['full']} ({info['league']}, {info['years']})")

    return "\n".join(lines)
