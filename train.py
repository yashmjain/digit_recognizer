# -*- coding: utf-8 -*-

from digit_recognizer_model import digit_recognizer,OUT_CHANNEL
from pre_processing import display_image
import numpy as np


def main():
    print ()
    dr = digit_recognizer()
    output = dr()
    for filter_no in range(OUT_CHANNEL):       
        display_image(output[0][filter_no].detach().numpy())
    

if __name__ == '__main__':
    main()

