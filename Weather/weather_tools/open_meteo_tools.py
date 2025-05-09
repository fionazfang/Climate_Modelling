from datetime import datetime
import requests

# URL for forecast
FORECAST_URL = "https://api.open-meteo.com/v1/forecast?"
GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search?"


def get_forecast(latitude, longitude, forecast_days):
    """
    Retrieves weather forecast information based on the provided latitude and
    longitude coordinates.

    This function queries a weather API to obtain weather forecast data for the
    specified location.
    The forecast includes current weather conditions and forecasted high
    temperatures for the next few days.

    Args:
        latitude: The latitude coordinate of the location.
        longitude: The longitude coordinate of the location.
        forecast_days: The number of forecast days to retrieve.

    Returns:
        dict: A dictionary containing the following keys:
            - 'current_temperature': The current temperature in Fahrenheit.
            - 'current_weather': A string representing the current weather condition.
            - 'high_temp_list': A list of forecasted high temperatures in Fahrenheit for each day.
            - 'date_list': A list of dates corresponding to the forecasted high temperatures.
            - 'wmo_list': A list of weather condition codes for each forecasted day.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": ["temperature_2m", "weather_code"],
        "daily": ["weather_code", "temperature_2m_max"],
        "temperature_unit": "fahrenheit",
        "wind_speed_unit": "mph",
        "precipitation_unit": "inch",
        "timezone": "auto",
        "forecast_days": forecast_days + 1,  # additional day for current
    }
    response = requests.get(FORECAST_URL, params=params).json()

    current_temperature = round(response["current"]["temperature_2m"])
    current_weather = str(response["current"]["weather_code"])
    high_temp_list = [
        round(temp) for temp in response["daily"]["temperature_2m_max"][1:]
    ]
    date_list = create_date_list(response)
    wmo_list = [str(code) for code in response["daily"]["weather_code"][1:]]

    return {
        "current_temperature": current_temperature,
        "current_weather": current_weather,
        "high_temp_list": high_temp_list,
        "date_list": date_list,
        "wmo_list": wmo_list,
    }


def get_cords_from_zip(zip):
    """
    Retrieves the city, latitude, longitude, and state for a given zip code.

    Args:
        zip (str): The zip code for which to retrieve the geographical coordinates.

    Returns:
        dict: A dictionary with the following keys:
            - 'latitude': The latitude of the city.
            - 'longitude': The longitude of the city.
            - 'city': The city corresponding to the zip code.
            - 'state': The state in which the city is located.
    """
    response = requests.get(f"{GEOCODING_URL}name={zip}").json()
    return {
        "latitude": response["results"][0]["latitude"],
        "longitude": response["results"][0]["longitude"],
        "city": response["results"][0]["name"],
        "state": response["results"][0]["admin1"],
    }


def create_date_list(forecast):
    """
    Converts a list of dates in the 'forecast' dictionary into a list of
    formatted date strings.

    Args:
        forecast (dict): A dictionary containing weather forecast data.
                         It is expected to have a 'daily' key containing a
                         'time' key, which is a list of date strings in the
                         format "%Y-%m-%d".

    Returns:
        list: A list of strings representing dates in the format "%a %d %b".
    """
    time_list = forecast["daily"]["time"]
    days = len(time_list)

    day_list = []
    for i in range(1, days):
        dt_object = datetime.strptime(time_list[i], "%Y-%m-%d")
        day_list.append(dt_object.strftime("%a %d %b"))
    return day_list
