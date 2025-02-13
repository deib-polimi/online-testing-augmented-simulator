import re

from domains.domain import WEATHERS, SEASONS, COUNTRIES, CITIES, LOCATIONS, TIMES

WEATHER_INSTRUCTIONS = [f"Make it {x}" for x in WEATHERS]
SEASON_INSTRUCTIONS = [f"Make it {x}" for x in SEASONS]
COUNTRY_INSTRUCTIONS = [f"Make it {x}" for x in COUNTRIES]
CITY_INSTRUCTIONS = [f"Make it {x}" for x in CITIES]
LOCATION_INSTRUCTIONS = [f"Make it {x}" for x in LOCATIONS]
TIME_INSTRUCTIONS = [f"Make it {x}" for x in TIMES]

ALL_INSTRUCTIONS = [
    *WEATHER_INSTRUCTIONS, *SEASON_INSTRUCTIONS, *COUNTRY_INSTRUCTIONS, *CITY_INSTRUCTIONS, *LOCATION_INSTRUCTIONS,
    *TIME_INSTRUCTIONS
]

WEATHER_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in WEATHERS}
SEASON_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in SEASONS}
COUNTRY_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in COUNTRIES}
CITY_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in CITIES}
LOCATION_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in LOCATIONS}
TIME_INSTRUCTIONS_MAP = {x: f"Make it {x}" for x in TIMES}

ALL_INSTRUCTIONS_MAP = {
    **WEATHER_INSTRUCTIONS_MAP, **SEASON_INSTRUCTIONS_MAP, **COUNTRY_INSTRUCTIONS_MAP, **CITY_INSTRUCTIONS_MAP,
    **LOCATION_INSTRUCTIONS_MAP, **TIME_INSTRUCTIONS_MAP
}

WEATHER_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in WEATHERS}
SEASON_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in SEASONS}
COUNTRY_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in COUNTRIES}
CITY_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in CITIES}
LOCATION_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in LOCATIONS}
TIME_INSTRUCTIONS_FOLDER_MAP = {x: re.sub('[^0-9a-zA-Z]+', '-', f"Make it {x}") for x in TIMES}

ALL_INSTRUCTIONS_FOLDER_MAP = {
    **WEATHER_INSTRUCTIONS_FOLDER_MAP, **SEASON_INSTRUCTIONS_FOLDER_MAP, **COUNTRY_INSTRUCTIONS_FOLDER_MAP,
    **CITY_INSTRUCTIONS_FOLDER_MAP, **LOCATION_INSTRUCTIONS_FOLDER_MAP, **TIME_INSTRUCTIONS_FOLDER_MAP
}

INSTRUCTION_TO_DOMAIN_MAP = {
    **{f"Make it {x}": x for x in WEATHERS},
    **{f"Make it {x}": x  for x in SEASONS},
    **{f"Make it {x}": x for x in COUNTRIES},
    **{f"Make it {x}": x  for x in CITIES},
    **{f"Make it {x}": x  for x in LOCATIONS},
    **{f"Make it {x}": x  for x in TIMES},
}