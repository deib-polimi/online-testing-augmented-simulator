import re

from domains.domain import WEATHERS, SEASONS, COUNTRIES, CITIES, LOCATIONS, TIMES

WEATHER_PROMPTS = [f"A street in {x} weather, photo taken from a car" for x in WEATHERS]
SEASON_PROMPTS = [f"A street in {x} season, photo taken from a car" for x in SEASONS]
COUNTRY_PROMPTS = [f"A street in {x}, photo taken from a car" for x in COUNTRIES]
CITY_PROMPTS = [f"A street in {x}, photo taken from a car" for x in CITIES]
LOCATION_PROMPTS = [f"A street in {x}, photo taken from a car" for x in LOCATIONS]
TIME_PROMPTS = [f"A street during {x}, photo taken from a car" for x in TIMES]
# WEATHER_PROMPTS = [f"A street in {x} weather" for x in WEATHERS]
# SEASON_PROMPTS = [f"A street in {x} season" for x in SEASONS]
# COUNTRY_PROMPTS = [f"A street in {x}" for x in COUNTRIES]
# CITY_PROMPTS = [f"A street in {x}" for x in CITIES]
# LOCATION_PROMPTS = [f"A street in {x}" for x in LOCATIONS]
# TIME_PROMPTS = [f"A street during {x}" for x in TIMES]

ALL_PROMPTS = [*WEATHER_PROMPTS, *SEASON_PROMPTS, *COUNTRY_PROMPTS, *CITY_PROMPTS, *LOCATION_PROMPTS, *TIME_PROMPTS]

WEATHER_PROMPTS_MAP = {x: f"A street in {x} weather, photo taken from a car" for x in WEATHERS}
SEASON_PROMPTS_MAP = {x: f"A street in {x} season, photo taken from a car" for x in SEASONS}
COUNTRY_PROMPTS_MAP = {x: f"A street in {x}, photo taken from a car" for x in COUNTRIES}
CITY_PROMPTS_MAP = {x: f"A street in {x}, photo taken from a car" for x in CITIES}
LOCATION_PROMPTS_MAP = {x: f"A street in {x}, photo taken from a car" for x in LOCATIONS}
TIME_PROMPTS_MAP = {x: f"A street in {x}, photo taken from a car" for x in TIMES}

ALL_PROMPTS_MAP = {
    **WEATHER_PROMPTS_MAP, **SEASON_PROMPTS_MAP, **COUNTRY_PROMPTS_MAP, **CITY_PROMPTS_MAP,
    **LOCATION_PROMPTS_MAP, **TIME_PROMPTS_MAP
}

WEATHER_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street in {x} weather, photo taken from a car") for x in WEATHERS}
SEASON_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street in {x} season, photo taken from a car") for x in SEASONS}
COUNTRY_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street in {x}, photo taken from a car") for x in COUNTRIES}
CITY_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street in {x}, photo taken from a car") for x in CITIES}
LOCATION_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street in {x}, photo taken from a car") for x in LOCATIONS}
TIME_PROMPTS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"A street during {x}, photo taken from a car") for x in TIMES}

ALL_PROMPTS_FOLDER_MAP = {
    **WEATHER_PROMPTS_FOLDER_MAP, **SEASON_PROMPTS_FOLDER_MAP, **COUNTRY_PROMPTS_FOLDER_MAP,
    **CITY_PROMPTS_FOLDER_MAP, **LOCATION_PROMPTS_FOLDER_MAP, **TIME_PROMPTS_FOLDER_MAP
}
