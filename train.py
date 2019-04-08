# -*- coding: utf-8 -*-

from digit_recognizer_model import digit_recognizer
from pre_processing import display_image
import numpy as np


def main():
    dr = digit_recognizer()
    output = dr()
    display_image(output.squeeze().squeeze().detach().numpy())
    

if __name__ == '__main__':
    main()

