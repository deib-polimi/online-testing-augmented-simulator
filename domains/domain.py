WEATHERS = ["cloudy", "dust storm", "foggy", "lightnings", "overcast", "smoke", "sunny"]
    # ["sunny", "cloudy", "rainy", "snowy", "thunderstorm", "hail", "foggy", "smoke", "dust storm", "lightnings",
    #         "windy", "blizzard", "freezing rain", "sleet", "high winds", "monsoon"]
SEASONS = ["spring", "summer", "autumn", "winter"]
COUNTRIES = ["usa", "china", "japan", "australia", "italy", "france", "canada", "germany", "india", "england",
             "brazil", "morocco"]
CITIES = ["new york", "san francisco", "chicago", "beijing", "tokyo", "sidney", "rome", "paris",
          "london", "toronto", "el cairo", "berlin"]
LOCATIONS = ["coastal area", "desert area", "mountain area", "rural area", "seaside area", "lake area", "forest area",
             "rivers", "plains"]
# TIMES = ["morning", "afternoon", "evening", "night", "sunrise", "sunset", "dawn", "dusk", "twilight"]
TIMES = ["morning", "afternoon", "evening", "night", "sunrise", "sunset", "dawn", "dusk"]

DOMAIN_CATEGORIES = [*WEATHERS, *SEASONS, *COUNTRIES, *CITIES, *LOCATIONS, *TIMES]

DOMAIN_CATEGORIES_MAP = {
    **{x: 'weather' for x in WEATHERS},
    **{x: 'seasons' for x in SEASONS},
    **{x: 'countries' for x in COUNTRIES},
    **{x: 'cities' for x in CITIES},
    **{x: 'locations' for x in LOCATIONS},
    **{x: 'times' for x in TIMES},
}
