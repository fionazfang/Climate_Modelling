from unittest.mock import patch, MagicMock

from project import get_location
from project import get_user_location
from project import print_weather_report


# Test the get_location function
@patch("requests.get")
def test_get_location(mock_get):
    # Mock API response
    mock_get.side_effect = [
        # ipify response
        type("Response", (), {"json": lambda: {"ip": "1.2.3.4"}}),
        # ip-api response
        type(
            "Response",
            (),
            {
                "json": lambda: {
                    "city": "TestCity",
                    "region": "TestRegion",
                    "country": "TestCountry",
                    "zip": "12345",
                }
            },
        ),
    ]

    expected_location_data = {
        "ip": "1.2.3.4",
        "city": "TestCity",
        "region": "TestRegion",
        "country": "TestCountry",
        "zip": "12345",
    }

    location_data = get_location()
    assert location_data == expected_location_data


# Test get_user_location function
@patch("project.get_location")
@patch("weather_tools.open_meteo_tools.get_cords_from_zip")
@patch("sys.exit")
def test_get_user_location(mock_exit, mock_get_cords, mock_get_location):
    # Testing current location
    mock_get_location.return_value = {"zip": "12345"}
    mock_get_cords.return_value = {
        "latitude": 123.456,
        "longitude": 123.456,
        "city": "New York City",
        "state": "New York",
    }

    args = MagicMock(zip=None)

    location_data, location_type = get_user_location(args)

    mock_get_cords.assert_called_with("12345")
    assert location_data == {
        "latitude": 123.456,
        "longitude": 123.456,
        "city": "New York City",
        "state": "New York",
    }
    assert location_type == " (Current Location)"
    mock_exit.assert_not_called()


# Test print_weather_report
def test_print_weather_report():
    user_location_details = {"city": "New York City", "state": "NY"}
    weather_report = {
        "current_temperature": 75,
        "current_weather": "3",
        "date_list": ["2024-05-15", "2024-05-16"],
        "high_temp_list": [80, 82],
        "wmo_list": ["0", "3"],
    }
    current_location = " (Current Location)"

    expected_output = (
        "\nWeather report: New York City, NY (Current Location)\n"
        "Currently: 75 degrees F\n"
        "           Cloudy\n"
        "┌──────────────────────────────────────────────────────────────────────┐\n"
        "│             2024-05-15 - High of 80 degrees F and Sunny              │\n"
        "└──────────────────────────────────────────────────────────────────────┘\n\n"
        "┌──────────────────────────────────────────────────────────────────────┐\n"
        "│             2024-05-16 - High of 82 degrees F and Cloudy             │\n"
        "└──────────────────────────────────────────────────────────────────────┘\n\n"
        "           ** For more options try 'python main.py --help' **           "
        "\n"
    )

    result = print_weather_report(
        user_location_details, weather_report, current_location
    )
    assert result == expected_output
