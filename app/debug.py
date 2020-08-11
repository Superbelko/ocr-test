"""Stuff that might be useful for debugging"""

def _rgb2bgr(color):
    return (color[2], color[1], color[0])

# some distinct colors, this list from http://phrogz.net/tmp/24colors.html
COLORS = [
    (255,0,0), #red
    (255,255,0), #yellow
    (0,234,255), #cyan
    (170,0,255), #magenta
    (255,127,0), #orange
    (191,255,0), #lime
    (0,149,255), #light blue
    (255,0,170), #pink
    (255,212,0), #gold
    (106,255,0), #light green
    (0,64,255), #blue
    (237,185,185), #pale red
    (185,215,237), #steel blue
    (231,233,185), #sand
    (220,185,237), #soft pink
    (185,237,224), #greenish light blue
    (143,35,35), #dark red
    (35,98,143), #ocean blue
    (143,106,35), #walnut
    (107,35,143), #dark violet
    (79,143,35), #grass green
    (115,115,115), #medium gray
    (204,204,204), #light gray
]

COLORS = list(map(_rgb2bgr, COLORS))
