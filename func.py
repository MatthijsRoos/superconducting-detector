'''
This module contains various miscellaneous functions used throughout SuperconductingDetector.
'''

import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline, RectBivariateSpline

def to_iterable_shape(data):
    '''
    Converts xdata/ydata/linekwargs to shape plotly() uses to plot individual or multiple traces.
    '''
    if isinstance(data, np.ndarray):
        return np.array(data, ndmin=2)
    elif isinstance(data, dict):
        return [data]
    elif not isinstance(data[0], (list, tuple, np.ndarray, dict, str)):
        return [data]
    return data



def findBW(xp, yp, extreme, xrange=[None,None], y_BW=-10):
    '''
    Finds full width at y=y_BW, i.e., the bandwidth (BW).
    
    xp: array
        data for x-axis
    yp: array
        data for y-axis
    extreme: int
        index argument for the extreme (bandwidth crosses
        on either side of extreme value)
    xrange: list, kwarg
        minimum and maximum indices for range to look for crossings
    y_BW: int or float, kwarg
        value of y at which we define the bandwidth, typically -10dB
        
    returns:
        float value of the found bandwidth
        array of the two frequencies at which crossings occur
    '''
    diff = np.abs(yp-y_BW)
    i_1, i_2 = np.argsort(diff[xrange[0]:extreme])[:2] + [0, xrange[0]][xrange[0]!=None]
    i_3, i_4 = np.argsort(diff[extreme:xrange[1]])[:2] + extreme
    
    x1 = inter_lin(xp[i_1], yp[i_1], xp[i_2], yp[i_2], y_BW)
    x2 = inter_lin(xp[i_3], yp[i_3], xp[i_4], yp[i_4], y_BW)
    
    # Unable to use the following, due to requirement of having no duplicate values for yp:
#     x1 = interp1D(y_BW, yp[xrange[0]:extreme], xp[xrange[0]:extreme])
#     x2 = interp1D(y_BW, yp[extreme:xrange[1]], xp[extreme:xrange[1]])
    return abs(x2 - x1), np.sort([x1, x2])



inter_lin = lambda x1,y1,x2,y2,y0: (y0-y1)*(x2-x1)/(y2-y1)+x1 # linear interpolation function

def interp1D(x, xp, yp):
    '''
    1D interpolator using scipy.interpolate.InterpolatedUnivariateSpline
    However, requires no duplicate values for xp.
    
    x: number or array
        coordinate(s) to interpolate at
    xp, yp: arrays
        xp and corresponding yp to use as data for interpolation, same lengths
    
    returns:
        interpolated y at x, same shape as x
    '''
    sorted_indices = np.argsort(xp)
    y = InterpolatedUnivariateSpline(np.array(xp)[sorted_indices], np.array(yp)[sorted_indices])
    return y(x)

def interp2D(x, y, xp, yp, zp):
    '''
    2D interpolator using scipy.interpolate.RectBivariateSpline
    However, requires no duplicate values for (xp,yp).
    
    x, y: numbers or arrays
        coordinate(s) to interpolate at
    xp, yp: arrays
        xp and corresponding yp to use as input data for interpolation
    zp: array
        matrix to use as data for interpolation, shape (len(x), len(y))
        
    returns:
        interpolated z at (x,y), shape (len(x), len(y))
    '''
    sorted_xindices = np.argsort(xp)
    sorted_yindices = np.argsort(yp)
    z = RectBivariateSpline(xp[sorted_xindices],yp[sorted_yindices],zp[sorted_xindices][:,sorted_yindices])
    return z(x, y)



def get_label(title, unit, flag=None, template=r"{}"):
    '''
    Used to format string for label titles.
    When using sd.plot.plotly(), the kwarg labels can be used; title and unit
    will be used to build appropriate axis labels.
    
    title: str
        The base for the label. Represents quantity of the axis
    unit: str
        The unit of the axis' quantity. If an empty string, no unit is added, and
        only the axis' quantity is returned
    flag: str or None, kwarg
        If not None, flag is returned
    template: str, kwarg
        String used for label formatting. By default, "title [unit]" is returned
        
    returns:
        str representing to be used as label title
    '''
    if flag is not None:
        return flag
    
    label = title
    
    if "mathrm" in template:
        space = "\:"
    else:
        space = " "
    
    if unit != "":
        label += space + r"[{}]".format(unit)
    
    return template.format(label)



def get_legend(legend):
    '''
    Used to format the legend dictionary to contain a combination of x, y, xanchor, and yanchor
    to correctly place the legend inside of the figure.
    If a dictionary is used as an input, it is simply passed on.
    '''
    if isinstance(legend, str):
        keys = legend.split(" ")
        location = dict(left   = dict(x=0.01, xanchor="left"),
                        center = dict(x=0.50, xanchor="center"),
                        right  = dict(x=0.99, xanchor="right"),
                        side   = dict(x=1.00, xanchor="left"),
                        bottom = dict(y=0.01, yanchor="bottom"),
                        middle = dict(y=0.50, yanchor="middle"),
                        top    = dict(y=0.99, yanchor="top"),
                        above  = dict(y=1.00, yanchor="bottom", orientation="h"))
        return dict(**location[keys[0]], **location[keys[1]])
    elif isinstance(legend, dict):
        return legend
    else:
        raise Exception("legend should either be a string containing location keywords, or a legend dictionary.")



def adaptive_tickvals(tickrange, step, insert):
    '''
    tickrange: list or tuple
        minimum and maximum, used for range
    step: number
        step used in placing ticks in tickrange
    insert: number
        value to insert in tickrange
    
    returns:
        tickvals inside range tickrange with insert added to it
    '''
    tickvals = np.arange(tickrange[0], tickrange[1]+step, step=step, dtype=float)
    tick_repl = np.argsort(np.abs(tickvals-insert))[0]
    repl = tickvals[tick_repl]

    if insert < repl:
        ins = [insert, repl+step/2]
    elif insert == repl:
        ins = [insert]
    else:
        ins = [repl-step/2, insert]
    
    dec = max([len(str(num).split(".")[-1]) for num in [step, insert]])
    return np.round([*tickvals[:tick_repl], *ins, *tickvals[tick_repl+1:]], dec)



def highlight(point, coldata, rowdata, **kwargs):
    '''
    Return dictionary containing data for graph_objects.layout.shape.
    Can be added to the kwarg shapes=[] and used to highlight points in, for example,
    plot.contour() or plot.heatmap()
    
    point: list or tuple
        contains coordinates of point to highlight
    coldata: list/array, kwarg
        list of array-like x-axis data
    rowdata: list/array, kwarg
        list of array-like y-axis data
    **kwargs:
        go to dictionary to further adjust layout of the shape
    '''
    coldata = np.array([2*coldata[0]-coldata[1], *coldata, 2*coldata[-1]-coldata[-2]])
    rowdata = np.array([2*rowdata[0]-rowdata[1], *rowdata, 2*rowdata[-1]-rowdata[-2]])
    carg, rarg = np.argmin(np.abs(coldata-point[0])), np.argmin(np.abs(rowdata-point[1]))
    return dict(x0=(coldata[carg]+coldata[carg-1])/2, x1=(coldata[carg]+coldata[carg+1])/2,
                y0=(rowdata[rarg]+rowdata[rarg-1])/2, y1=(rowdata[rarg]+rowdata[rarg+1])/2,
                type="rect", **kwargs)