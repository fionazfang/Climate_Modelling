import json


def make_basic_wmo():
    wmo_dict = {}

    with open('wmo_descriptions.json') as json_file:
        data = json.load(json_file)
        # print(json.dumps(data, indent=2))
        for wmo in data.keys():
            # print(wmo, data[wmo]['day']['description'])
            wmo_dict[wmo] = data[wmo]['day']['description']

    print(wmo_dict)

    with open('wmo_basic.json', 'w') as json_file:
        json.dump(wmo_dict, json_file)


def json_testing():
    with open('wmo_basic.json') as json_file:
        wmo_descriptions = json.load(json_file)

    weather_code = '0'
    print(wmo_descriptions[weather_code])


def main():
    json_testing()


if __name__ == "__main__":
    main()
