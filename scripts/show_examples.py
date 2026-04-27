import json
import random

with open("data/tunes.json") as f:
    tunes = json.load(f)

samples = random.sample(tunes, 5)
print(json.dumps(samples, indent=2))
