WEATHERS = ["sunny", "cloudy", "rainy", "snowy", "thunderstorm", "hail", "foggy", "smoke", "dust storm", "lightnings",
            "windy", "blizzard", "freezing rain", "sleet", "high winds", "monsoon"]
SEASONS = ["spring", "summer", "autumn", "winter"]
COUNTRIES = ["usa", "china", "japan", "australia", "italy", "france", "canada", "germany", "india", "england",
             "switzerland", "brazil", "argentina", "egypt", "greece", "morocco", "russia", "south korea", "mexico",
             "spain", "netherlands", "indonesia"]
CITIES = ["new york", "san francisco", "chicago", "los angeles", "beijing", "tokyo", "sidney", "rome", "paris",
          "london", "toronto", "el cairo", "berlin", "munich", "moscow", "dubai", "singapore", "instanbul"]
LOCATIONS = ["coastal area", "desert area", "mountain area", "rural area", "seaside area", "lake area", "forest area",
             "island towns", "wetlands", "rivers", "plains", "canyons", "tundra"]
TIMES = ["morning", "afternoon", "evening", "night", "sunrise", "sunset", "dawn", "dusk", "twilight"]

DOMAIN_CATEGORIES = [*WEATHERS, *SEASONS, *COUNTRIES, *CITIES, *LOCATIONS, *TIMES]

DOMAIN_CATEGORIES_MAP = {
    **{x: 'weather' for x in WEATHERS},
    **{x: 'seasons' for x in SEASONS},
    **{x: 'countries' for x in COUNTRIES},
    **{x: 'cities' for x in CITIES},
    **{x: 'locations' for x in LOCATIONS},
    **{x: 'times' for x in TIMES},
}
