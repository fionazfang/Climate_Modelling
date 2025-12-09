import argparse
import json
import requests
import sys

from weather_tools import open_meteo_tools as omt


def main():
    """
    Parses command line arguments for a US zip code and number of forecast days,
    retrieves the user's location details, gets the weather forecast for the
    specified number of days, and prints the weather report to the screen.

    Command line arguments:
        -z, --zip: An integer representing a US zip code.
        -d, --days: An integer between 1 and 15 representing the number of days
                    to show the forecast for. Default is 3.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-z", "--zip", type=int, help="search by US zipcode")
    parser.add_argument(
        "-d",
        "--days",
        type=int,
        choices=range(1, 16),
        default=3,
        metavar="DAYS",
        help="number of days to show forecast between 1 and 15 (default: 3)",
    )
    args = parser.parse_args()

    user_location_details, current_location = get_user_location(args)

    weather_report = omt.get_forecast(
        user_location_details["latitude"],
        user_location_details["longitude"],
        forecast_days=args.days,
    )

    # Print weather report to screen
    print(print_weather_report(user_location_details, weather_report, current_location))


def get_user_location(args):
    """
    Retrieves the user's location based on the provided zip code or the current
    location if no zip code is provided.

    Args:
        args (argparse.Namespace): Command line arguments parsed by argparse.
                                   It is expected to have a 'zip' attribute.

    Returns:
        tuple: A tuple containing a dictionary with the user's location details
        and a string indicating whether the location is the current location or
        a user-specified location.

    Raises:
        SystemExit: If an invalid zip code is provided.
    """
    if args.zip == None:
        # Find location from blank command prompt
        user_location = get_location()
        return omt.get_cords_from_zip(user_location["zip"]), " (Current Location)"
    else:
        # Find location from user defined zip code
        try:
            return omt.get_cords_from_zip(args.zip), ""
        except KeyError:
            print("Error: Invalid ZIP code provided.")
            sys.exit(1)


def get_location():
    """
    Retrieves the location data of the user based on their IP address.

    This function first retrieves the user's IP address using the ipify API.
    Then, it queries the ip-api.com service to get detailed location information
    based on the IP address.

    Returns:
        dict: A dictionary with the following keys:
            - 'ip': The IP address of the user.
            - 'city': The city where the IP is located.
            - 'region': The region where the IP is located.
            - 'country': The country where the IP is located.
            - 'zip': The zip code where the IP is located.
    """
    ip_address = requests.get("https://api64.ipify.org?format=json").json()["ip"]
    response = requests.get(f"http://ip-api.com/json/{ip_address}").json()
    location_data = {
        "ip": ip_address,
        "city": response.get("city"),
        "region": response.get("region"),
        "country": response.get("country"),
        "zip": response.get("zip"),
    }
    return location_data


def print_weather_report(user_location_details, weather_report, current_location):
    """
    Prints a formatted weather report based on the provided location and weather data.

    This function takes the user's location details, weather report data, and an
    indication of whether the report is for the current location. It then formats
    and prints the weather report in a visually appealing manner.

    Args:
        user_location_details (dict): A dictionary containing the user's location
            details, including 'city' and 'state'.
        weather_report (dict): A dictionary containing the weather report data,
            including forecasted high temperatures, dates, and weather descriptions.
        current_location (bool): Indicates whether the report is for the current location.

    Returns:
        Weather report as concatenated string
    """
    text_width = 72
    day_count = len(weather_report["date_list"])

    # Load json with wmo code descriptions
    with open("./weather_tools/wmo_basic.json") as json_file:
        wmo_descriptions = json.load(json_file)

    # Start with blank string
    weather_report_return = ""
    weather_report_return += "\n"
    weather_report_return += f"Weather report: {user_location_details['city']}, {user_location_details['state']}{current_location}\n"
    weather_report_return += f"Currently: {weather_report['current_temperature']} degrees F\n           {wmo_descriptions[weather_report['current_weather']]}\n"

    for i in range(day_count):
        date = weather_report["date_list"][i]
        high_temp = weather_report["high_temp_list"][i]
        weather_code = weather_report["wmo_list"][i]
        weather_description = wmo_descriptions[weather_code]
        forecast_text = (
            "│"
            + f"{date} - High of {high_temp} degrees F and {weather_description}".center(
                text_width - 2
            )
            + "│\n"
        )

        weather_report_return += f"┌{'─' * (text_width - 2)}┐\n"
        weather_report_return += forecast_text
        weather_report_return += f"└{'─' * (text_width - 2)}┘\n\n"

    if current_location:
        weather_report_return += (
            "** For more options try 'python main.py --help' **".center(text_width)
        )
        weather_report_return += "\n"
    return weather_report_return


if __name__ == "__main__":
    main()
