'''
This module contains simple ways of creating and showing color gradients for the use in plotting functions found in plot.py
'''

import numpy as np

def default():
    return ["#636EFA", # Royal Blue
            "#EF553B", # Tomato Red
            "#32CD32", # Lime Green
            "#FF8C00", # Dark Orange
            "#008080", # Teal
            "#8A2BE2", # Blue Violet
            "#C71585", # Medium Violet Red
            "#1A233D", # Midnight Blue
           ]

def SRON():
    return ["#D53F15", # Orange 1
            "#EB5D34", # Orange 2
            "#181932", # Blue
            "#FFFFFF", # White
            "#BE6D65", # Tone 1
            "#CECFDA", # Tone 2
            "#E5E6F3", # Tone 3
           ]

def DCD():
    return ["#00528C", # DCD blue
            "#00A6D8", # DCD AT blue
            "#75B843", # DCD OD green
            "#EB5E57", # DCD CD red
            "#F7AB23", # DCD IHD yellow
            "#E9EEF9", # DCD white
            "#001A3B", # DCDark blue
           ]

def mapscale():
    return ['#1C1E34',
            '#24284A',
            '#2F3460',
            '#3B4375',
            '#49548A',
            '#5D678F',
            '#7A757F',
            '#A28F87',
            '#CDB8AC',
            '#E6DCD6'
           ]

inter_color = lambda c1, c2, t: tuple(c1[i] + (c2[i] - c1[i]) * t for i in range(3))

def simplegradient(n, start=default()[0], end=default()[1]):
    '''
    To create a list of length n containing hex codes defining a color gradient.
    '''
    from matplotlib import colors
    c_start = colors.to_rgb(start)
    c_end = colors.to_rgb(end)
    return [start]+[colors.to_hex(inter_color(c_start, c_end, (i+1)/n)) for i in range(n-2)]+[end]

def gradient_middle(n, start=default()[0], end=default()[1], middle="#ffffff"):
    '''
    To create a list of length n containing hex codes defining a color gradient. 
    Defaults to Royal Blue-->white-->Tomato Red
    '''
    if n//2 == n/2:
        return simplegradient(n//2,   start=start, end=middle) + simplegradient(n//2,   start=middle, end=end)
    else:
        return simplegradient(n//2+1, start=start, end=middle) + simplegradient(n//2+1, start=middle, end=end)[1:]

def gradient_replacedmiddle(n, start=default()[0], end=default()[1], middle="#ffffff", replace=default()[2]):
    '''
    To create a list of length n containing hex codes defining a color gradient with its center replaced. 
    Defaults to Royal Blue-->white-->Tomato Red, where white gets replaced by Lime green
    '''
    if isinstance(replace, str):
        replace = [replace]
    elif isinstance(replace, (list, tuple, np.ndarray)):
        n -= len(replace)
        replace = list(replace)
    else:
        raise Exception("Check type of replace")
    
    gradient_before = simplegradient(n//2+1, start=start, end=middle)[:-1]
    gradient_after  = simplegradient(n//2+1, start=middle, end=end)[1:]
    return gradient_before + replace + gradient_after

def mapscale_midpoint(n=11, start=SRON()[2], end=SRON()[0], middle=SRON()[5], replace=DCD()[4]):
    '''
    To create a list of length n containing hex codes defining a color gradient with its center replaced.
    '''
    return gradient_replacedmiddle(n, start, end, middle, replace)

def printcolors(colors):
    print("colors = [",*["'"+c+"'," for c in colors],"]", sep='\n')

def showcolors(colors):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    for i, color in enumerate(colors):
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
    ax.set_xlim(0, len(colors))
    ax.set_ylim(0, 1)
    ax.axis("off")
    
def sampleColorway(vals, colors=mapscale()):
    '''
    To convert a set of values to a list of samples using a list of colors or a built-in colorscale. 
    Linearly scales the values in vals between [0,1] and assigns samples from 'colors'.
    '''
    import plotly.colors
    return plotly.colors.sample_colorscale(colors, (vals-vals[0])/(vals[-1]-vals[0]))
    
def sampleColorscale(vals, colors=mapscale()):
    '''
    To convert a set of values to a colorscale using samples of a list of colors. 
    Linearly scales the values in vals between [0,1] and assigns samples from 'colors'.
    '''
    colors_samples = sampleColorway(vals, colors)
    return list(zip( (vals-vals[0])/(vals[-1]-vals[0]),  colors_samples ))
    
def toColorscale(colors=mapscale(), mid=0.5):
    '''
    To convert a set of colors to a colorscale.
    By changing mid to a different value in (0,1), 
    one can rescale the center color value to be at that new midpoint
    '''
    n = len(colors)
    vals = np.arange(n)/(n-1)*2*mid
    vals[vals>mid] = vals[vals>mid]*(1-mid)/(mid) + 2*mid-1
    return list(zip(vals, colors))

def repeatcolors(n, colors=default()):
    '''
    Convert colors to have each element repeated n times.
    '''
    return [c for c in colors for _ in range(n)]
