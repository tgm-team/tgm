import csv
import tempfile

import torch

from opendg._io.tgb import read_tgb
from opendg.events import EdgeEvent


def test_tgb_conversion_no_features():
    name = "tgbl-wiki"
    read_tgb(name=name)



def main():
    
    test_tgb_conversion_no_features()

if __name__ == "__main__":
    main()