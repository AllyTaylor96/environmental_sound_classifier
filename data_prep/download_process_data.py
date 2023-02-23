"""Need to convert this to dataset downloading, formatting etc."""

import os
import json
import urllib.request


with open('../config.json', 'r') as f:
    config = json.load(f)


response = urllib.request.urlretrieve('https://github.com/karoldvl/ESC-50/archive/master.zip')

