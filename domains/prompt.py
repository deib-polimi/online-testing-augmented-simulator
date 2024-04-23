from domains.domain import WEATHERS, SEASONS, COUNTRIES, CITIES, LOCATIONS, TIMES

WEATHER_PROMPTS = [f"A street in {x} weather, photo taken from a car" for x in WEATHERS]
SEASON_PROMPTS = [f"A street in {x} season, photo taken from a car" for x in SEASONS]
COUNTRY_PROMPTS = [f"A street in {x}, photo taken from a car" for x in COUNTRIES]
CITY_PROMPTS = [f"A street in {x}, photo taken from a car" for x in CITIES]
LOCATION_PROMPTS = [f"A street in {x}, photo taken from a car" for x in LOCATIONS]
TIME_PROMPTS = [f"A street during {x}, photo taken from a car" for x in TIMES]

ALL_PROMPTS = [*WEATHER_PROMPTS, *SEASON_PROMPTS, *COUNTRY_PROMPTS, *CITY_PROMPTS, *LOCATION_PROMPTS, *LOCATION_PROMPTS]
