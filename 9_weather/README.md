# CS50P Final Project - Command Line Weather Forecast

#### Video Demo: https://youtu.be/YWQlrteoJn0

#### Description

A simple command line program that shows the weather at the current location and
a brief three-day forecast.  This project allowed me to demonstrate various
Python methods including command line arguments, API calls, and creating and
referencing my own libraries to separate and organize the code.

## Overview

Inspired by the popular "wttr.in" project that returns the weather via a curl or
wget, I chose to build a basic version for command-line use.  One thing I wanted
to incorporate was a "default" forecast with the current user's location.  I
accomplished this by first finding the user's IP address via "ipify.org" and then
utilizing the returned IP to find the location via "ip-api.com".  For weather
forecasting, I debated between Openweathermap and Open Meteo, but ultimately the
need for not needing a key with Open Meteo outweighed having fewer search
features.

## Features

Utilizing the Argparse library was an easy way to incorporate both a standard
help menu and basic error checking of the user-given arguments before being
processed further in the code.

## Technologies Used

Most of the project just utilizes the Requests library and various API calls for
the location of the user based on IP address and the weather forecast from
Open-Meteo.

### Prerequisites

Python 3.12.2 was installed in VS Codespaces at the time of the project.

### Installation

See "requirements.txt" for needed Python libraries.  Recommend that a virtual
environment is created after which the following command:
```
pip install -r requirements.txt
````
Once the additional libraries have been installed start the server with the
command:
```
python project.py
```

## Usage

Running "python main.py" checks the weather at the current location and the
default three-day forecast.  The user has the option to choose a location or
forecast length with additional command line arguments:

The various command line arguments can also be used:
```
python project.py --zip (or -z)
```
for looking up the weather in a specific location or:
```
python project.py --days (or -d)
```
for adjusting the number of returned days of the forecast.  These arguments can
be used together or independently which allows the user to customize the
information returned.

## Roadmap

 - Add more information to the forecast, i.e. low temp, wind speed/direction
 - Add a "search" type argument for the user to use City, State
 - Potentially publish the open_meteo_tools library for others to use.

## Acknowledgements

wttr.in — the right way to check curl the weather! -
https://github.com/chubin/wttr.in

Open-Meteo - https://open-meteo.com/

ipify API - https://www.ipify.org/

ip-api - https://ip-api.com/

## Contact

John Baumert - <baumert.john@gmail.com> - <https://www.johnbaumert.com/>
