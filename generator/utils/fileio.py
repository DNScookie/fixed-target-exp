import datetime
import json


def to_file(sim_dict, filename: str = f'data/simulation_{datetime.datetime}.json'):
    with open(filename, mode="w") as f:
        f.write(json.dumps(sim_dict))


def from_file(filename: str):
    with open(filename, "r") as f:
        return json.load(f)
