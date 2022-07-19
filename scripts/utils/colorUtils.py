from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color
from colormath.color_diff import delta_e_cie2000

from typing import Tuple

def getColorDifference(c1: Tuple[int, int, int], c2: Tuple[int, int, int]) -> float:
    color1Rgb = sRGBColor(c1[0] / 255.0, c1[1] / 255.0, c1[2] / 255.0)
    color2Rgb = sRGBColor(c2[0] / 255.0, c2[1] / 255.0, c2[2] / 255.0)
    color1Lab = convert_color(color1Rgb, LabColor)
    color2Lab = convert_color(color2Rgb, LabColor)
    return delta_e_cie2000(color1Lab, color2Lab)
