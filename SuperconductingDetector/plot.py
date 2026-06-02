'''
This module contains several functions to ease plotting using Plotly. plotly() contains standard methods to plot x- and y-data.
'''

import numpy as np
import plotly.graph_objects as go
from . import func, color

def plotly(xdata, ydata, IDs=None, savename=None, hlines=[], vlines=[], x0=None, show_BW=False, y_BW=-10,
           xrange=[None,None], yrange=[None,None], xlabel=None, ylabel=None, size=(500,400), legend="top left",
           labels=dict(x="x", y="y", x_unit="", y_unit=""), label_template=r"$\mathrm{{{}}}$", colorway=color.default(), 
           linekwargs=dict(), template='base+light', add_traces=[], return_traces=False, show_annotations=False, sig=3, 
           **kwargs):
    '''
    Plots data and determines at which y-level x=x0 is crossed.
    
    xdata: list/array or list/array of lists/arrays
        list of array-like x-axis data
    ydata: list/array or list/array of lists/arrays
        list of array-like y-axis data
    IDs: string or list of strings, kwarg
        string, list of strings, or return of writeCSV().
        Length has to match number of data sets provided
    savename: string, kwarg
        file name for saving
    hlines: list, kwarg
        list of y-values at which horizontal lines should be added
    vlines: list, kwarg
        list of x-values at which vertical lines should be added
    x0: int or float, kwarg
        if not None, x0 is used in plot to show at which y0
        the data crosses x=x0. y0 is added to legend
    show_BW: bool, kwarg
        if True, show bandwidth in plot and add to legend. 
        Used if x0 is not None. x0 taken as being close to extreme value
    y_BW: int or float, kwarg
        value of y at which the bandwidth is defined, typically -10dB or -3dB. 
        Passed on to func.findBW(). Used if showBW=True
    xrange: list, kwarg
        minimum and maximum for range xaxis
    yrange: list, kwarg
        minimum and maximum for range yaxis
    xlabel: str, kwarg
        string as title for x-axis. If xlabel is None, it is built using labels
    ylabel: str, kwarg
        string as title for y-axis. If ylabel is None, it is built using labels
    size: tuple, kwarg
        tuple containing (width, height) for size of figure
    legend: str or dict, kwarg
        either a string used to position the legend in the figure using get_legend()
        or a dictionary used as input in layout
    labels: dict, kwarg
        dictionary containing strings to use for quantities and their units on x and y axes.
        Used for string building for the legend or the axes' labels
    label_template: str, kwarg
        String used for label formatting. Passed on to sd.func.get_label()
    colorway: list, kwarg
        list containing color code strings. Use SuperConductor.colors to easily
        create color lists for colorway
    linekwargs: dict or list of dicts, kwarg
        list containing dictionaries to be iterated as kwargs for each trace
    template: str, kwarg
        string of go.layout.Template
    add_traces: list or trace, kwarg
        list of trace specifications used as data for add_traces()
    return_traces: bool, kwarg
        if True, plotly() returns all traces
    show_annotations: bool, kwarg
        if True, show annotations for each hlines and vline drawn. Includes x0 and BW vlines.
    sig: number, kwarg
        Number of significant digits used in legend and annotations. Default is 3
    **kwargs:
        go to update_layout()
        
    returns:
        go.Figure() 
        OR list containing traces if return_traces
    '''
    xdata = func.to_iterable_shape(xdata)
    ydata = func.to_iterable_shape(ydata)
    
    linekwargs = func.to_iterable_shape(linekwargs)
    if len(linekwargs) == 1:
        linekwargs *= len(xdata)
    
    if IDs is None:
        IDs = [""]*len(xdata)
        showlegend = x0 is not None
    else:
        showlegend = True
    IDs = np.array(IDs, dtype=str, ndmin=1)
    
    if not (len(xdata) == len(ydata) == len(IDs) == len(linekwargs)):
        print("Some traces don't appear as one or more of xdata, ydata, IDs, or linekwargs has less elements than the other input data.")
    
    # Layout settings
    xlog, ylog = ("xlog" in template.split("+")), ("ylog" in template.split("+"))
    xlabel = func.get_label(labels["x"], labels["x_unit"], xlabel, label_template)
    ylabel = func.get_label(labels["y"], labels["y_unit"], ylabel, label_template)
    line   = dict(color='black' if np.any(np.array(template.split("+")) == "light") else 'white', dash='dash', width=1.2)
    legend = func.get_legend(legend)
    fmt    = f"{{:.{sig}g}}"
    
    fig = go.Figure(layout = dict(template=template, width=size[0], height=size[1], showlegend=showlegend,
                                  xaxis = dict(title=xlabel, range=xrange, spikecolor=line["color"],
                                               spikedash=line["dash"], spikethickness=line["width"]), 
                                  yaxis = dict(title=ylabel, range=yrange), hovermode="x unified", 
                                  legend=legend, **kwargs))
    
    # Add vertical and horizontal guidelines
    if x0 is not None:
        vlines = np.append(vlines, x0)
    for v in vlines:
        fig.add_vline(x=v, line=line, annotation_text=fmt.format(v) if show_annotations else "", 
                      annotation_position='top', annotation_x=v if not xlog else np.log10(v))
    for h in hlines:
        fig.add_hline(y=h, line=line, annotation_text=fmt.format(h) if show_annotations else "",
                      annotation_position='right', annotation_y=h if not ylog else np.log10(h))
    
    # Add all traces
    traces = []
    for i, (x, y, ID, lkwargs) in enumerate(zip(xdata, ydata, IDs, linekwargs)): # loop over all input data
        color = colorway[i%len(colorway)]
        name = ID
        if x0 is not None:
            y0 = func.interp1D(x0, x, y)
            name += labels["y"] +"("+ labels["x"] +"="+ fmt.format(x0) + labels["x_unit"] +") = "+ fmt.format(y0) + labels["y_unit"]
            
            if show_BW:
                argmin = np.argmin(np.abs(x-xrange[0])) if xrange[0] is not None else None
                argmax = np.argmin(np.abs(x-xrange[1])) if xrange[1] is not None else None
                BW, xvals = func.findBW(x, y, np.argmin(np.abs(x-x0)), [argmin, argmax], y_BW)
                for x_BW in xvals:
                    fig.add_vline(x=x_BW, line=dict(dash='dash', width=1, color=color),
                                  annotation_text=fmt.format(x_BW) if show_annotations else "", annotation_position='top',
                                  annotation_font_color=color, annotation_x=x_BW if not xlog else np.log10(x_BW))
                name += "; BW = " + fmt.format(BW) + labels["x_unit"]
        
        traces.append(go.Scatter(x=x, y=y, hovertemplate = ID + ": %{y}" + labels["y_unit"] + "<extra></extra>",
                                 name=name, line_color=color, **lkwargs))
    if return_traces:
        return traces
    else:
        fig.add_traces(traces+add_traces)

        fig.show()
        if savename is not None:
            save_figure(fig, savename, size, template)



def save_figure(fig, savename, size, template, **savekwargs):
    '''
    Used to save figures made with plotting functions from plot.py
    '''
    if savename.split(".")[-1] == "html":
        savefunction = fig.write_html
        js = '''document.body.style.backgroundColor = "#000"; '''
        savekwargs.update(dict(include_plotlyjs="cdn", include_mathjax='cdn', auto_open=True,
                               post_script = js if np.any(np.array(template.split("+")) == "dark") else None))
    else:
        savefunction = fig.write_image
        savekwargs.update(dict(width=size[0], height=size[1], scale=1))
    savefunction(f"{savename}", **savekwargs)



def heatmap(data, coldata=None, rowdata=None, savename=None, size=(500,500),
            xlabel=None, ylabel=None, zlabel=None, xrange=[None,None], yrange=[None,None], zrange=[None,None],
            labels=dict(z="", col="x", row="y", z_unit="", col_unit="", row_unit=""), label_template=r"$\mathrm{{{}}}$", zsmooth=False,
            colorscale=color.toColorscale(), reversescale=False, template="base", hovertemplate=None, traces=[], **kwargs):
    '''
    Plots data in a heatmap.
    
    data: list/array (matrix)
        matrix containing z-data
    coldata: list/array, kwarg
        list of array-like x-axis data
    rowdata: list/array, kwarg
        list of array-like y-axis data
    savename: string, kwarg
        file name for saving
    size: tuple, kwarg
        tuple containing (width, height) for size of figure
    xrange: list, kwarg
        minimum and maximum for range x-axis
    yrange: list, kwarg
        minimum and maximum for range y-axis
    zrange: list, kwarg
        minimum and maximum for range z-axis
    xlabel: str, kwarg
        string as title for x-axis. If xlabel is None, it is built using labels
    ylabel: str, kwarg
        string as title for y-axis. If ylabel is None, it is built using labels
    zlabel: str, kwarg
        string as title for z-axis. If zlabel is None, it is built using labels
    labels: dict, kwarg
        dictionary containing strings to use for quantities and their units on x, y, and z axes.
        Used for string building for the axes' labels
    label_template: str, kwarg
        String used for (x- and y-)label formatting. Passed on to sd.func.get_label()
    zsmooth: str, kwarg
        Used as input for zsmooth in go.Heatmap(). Valid inputs: False for no z-smoothing;
        "fast" for 1d interpolation; "best" for bicubic interpolation
    colorscale: list or string, kwarg
        colorscale input used in go.Heatmap
    reversescale: bool, kwarg
        if True, reverses used colorscale
    template: str, kwarg
        string of go.layout.Template
    hovertemplate: str, kwarg
        if not None, string used as hovertemplate input in go.Heatmap
    traces: list or trace, kwarg
        list of trace specifications used as data for add_traces()
    **kwargs:
        go to update_layout()
        
    returns:
        go.Figure()
    '''
    xlabel = func.get_label(labels["col"], labels["col_unit"], xlabel, label_template)
    ylabel = func.get_label(labels["row"], labels["row_unit"], ylabel, label_template)
    zlabel = func.get_label(labels["z"],   labels["z_unit"], zlabel)
    hovertemplate = hovertemplate if hovertemplate is not None\
                                  else labels["z"]  +": %{z}" + labels["z_unit"]  +"<br>" +\
                                       labels["col"]+": %{x}" + labels["col_unit"]+"<br>" +\
                                       labels["row"]+": %{y}" + labels["row_unit"]+"<extra></extra>"
                                       
    line = dict(color='black' if np.any(np.array(template.split("+")) == "light") else 'white', dash='dash', width=1.2)

    fig = go.Figure(go.Heatmap(x=coldata, y=rowdata, z=data, colorbar_title=zlabel, zmin=zrange[0], zmax=zrange[1],
                               zsmooth=zsmooth, colorscale=colorscale, reversescale=reversescale, hovertemplate=hovertemplate),
                    layout = dict(xaxis = dict(title=xlabel, range=xrange, spikecolor=line["color"],
                                               spikedash=line["dash"], spikethickness=line["width"]), 
                                  yaxis = dict(title=ylabel, range=yrange, spikecolor=line["color"],
                                               spikedash=line["dash"], spikethickness=line["width"]),
                                  template=template, width=size[0]+46, height=size[1], **kwargs))
    fig.add_traces(traces)
    fig.show()
    if savename is not None:
        save_figure(fig, savename, (size[0]+46, size[1]), template)



def contour(data, coldata=None, rowdata=None, savename=None, size=(500,500),
            xlabel=None, ylabel=None, zlabel=None, xrange=[None,None], yrange=[None,None], zrange=[None,None],
            labels=dict(z="", col="x", row="y", z_unit="", col_unit="", row_unit=""), label_template=r"$\mathrm{{{}}}$",
            colorscale=color.toColorscale(), reversescale=False, template="base", hovertemplate=None, traces=[], **kwargs):
    '''
    Plots data in a contour plot.
    
    data: list/array (matrix)
        matrix containing z-data
    coldata: list/array, kwarg
        list of array-like x-axis data
    rowdata: list/array, kwarg
        list of array-like y-axis data
    savename: string, kwarg
        file name for saving
    size: tuple, kwarg
        tuple containing (width, height) for size of figure
    xrange: list, kwarg
        minimum and maximum for range x-axis
    yrange: list, kwarg
        minimum and maximum for range y-axis
    zrange: list, kwarg
        minimum and maximum for range z-axis
    xlabel: str, kwarg
        string as title for x-axis. If xlabel is None, it is built using labels
    ylabel: str, kwarg
        string as title for y-axis. If ylabel is None, it is built using labels
    zlabel: str, kwarg
        string as title for z-axis. If zlabel is None, it is built using labels
    labels: dict, kwarg
        dictionary containing strings to use for quantities and their units on x, y, and z axes.
        Used for string building for the axes' labels
    label_template: str, kwarg
        String used for (x- and y-)label formatting. Passed on to sd.func.get_label()
    colorscale: list or string, kwarg
        colorscale input used in go.Contour
    reversescale: bool, kwarg
        if True, reverses used colorscale
    template: str, kwarg
        string of go.layout.Template
    hovertemplate: str, kwarg
        if not None, string used as hovertemplate input in go.Contour
    traces: list or trace, kwarg
        list of trace specifications used as data for add_traces()
    **kwargs:
        go to update_layout()
        
    returns:
        go.Figure()
    '''
    xlabel = func.get_label(labels["col"], labels["col_unit"], xlabel, label_template)
    ylabel = func.get_label(labels["row"], labels["row_unit"], ylabel, label_template)
    zlabel = func.get_label(labels["z"],   labels["z_unit"], zlabel)
    hovertemplate = hovertemplate if hovertemplate is not None\
                                  else labels["z"]  +": %{z}" + labels["z_unit"]  +"<br>" +\
                                       labels["col"]+": %{x}" + labels["col_unit"]+"<br>" +\
                                       labels["row"]+": %{y}" + labels["row_unit"]+"<extra></extra>"
                                       
    line = dict(color='black' if np.any(np.array(template.split("+")) == "light") else 'white', dash='dash', width=1.2)

    fig = go.Figure(go.Contour(x=coldata, y=rowdata, z=data, colorbar_title=zlabel, zmin=zrange[0], zmax=zrange[1],
                               colorscale=colorscale, reversescale=reversescale, hovertemplate=hovertemplate),
                    layout = dict(xaxis = dict(title=xlabel, range=xrange, spikecolor=line["color"],
                                               spikedash=line["dash"], spikethickness=line["width"]), 
                                  yaxis = dict(title=ylabel, range=yrange, spikecolor=line["color"],
                                               spikedash=line["dash"], spikethickness=line["width"]),
                                  template=template, width=size[0]+46, height=size[1], **kwargs))
    fig.add_traces(traces)
    fig.show()
    if savename is not None:
        save_figure(fig, savename, (size[0]+46, size[1]), template)

        
        
def sweepZ0(network, Z0, vlines=[], f0=70, frange=[None,None], size=(800,600), savename=None, crop=False, **kwargs):
    '''
    Used to sweep a network in port impedance Z0
    
    network: object of class skrf.Network
        Antenna network
    Z0: number or array
        Port impedance(s) to evaluate, in Ohm
    vlines: list, kwarg
        list of frequency values at which vertical lines should be added
    f0: int or float, kwarg
        if not None, f0 is used in plot to show at which y0
        the data crosses f=f0. y0 is added to legend
    frange: list, kwarg
        minimum and maximum for range xaxis
    size: tuple, kwarg
        tuple containing (width, height) for size of figure(s)
    savename: string, kwarg
        file name for saving
    crop: bool, kwarg
        if True, crops the network data to the frequency range provided (frange)
    **kwargs:
        go to update_layout() in plotly()
        
    returns:
        1 go.Figure() from plotly() for Re{Z11} and Im{Z11}
        1 go.Figure() from plotly() for S11(Z0)
    '''
    import skrf as rf
    
    Z0 = np.array(Z0, ndmin=1) # Ohm (rescale to this port impedance)
    
    if crop:
        network = network[f'{frange[0]}-{frange[1]}ghz']
    f = network.frequency.f_scaled # GHz

    plotly([f]*2, [network.z_re.flatten(), network.z_im.flatten()], ["Re", "Im"],\
           x0=f0, vlines=vlines, hlines=[0], xrange=frange, size=size,\
           title_text="This plot is not saved" if savename is not None else network.name+" Impedance",\
           labels=dict(x="f", y="{Z11}", x_unit="GHz", y_unit="\u2126"), ylabel='$\mathrm{Z_{11}\:[\u2126]}$')

    S11s, IDs = [], []
    for Z in Z0:
        network.renormalize(Z)
        S11s.append(network.s_db.flatten())
        IDs.append(f"Z0 = {str(Z): >3}\u2126")

    plotly([f]*len(Z0), S11s, IDs=IDs, xrange=frange, yrange=[None,0], vlines=vlines, hlines=[-10], size=size, savename=savename,\
           labels=dict(x="f", y="S11", x_unit="GHz", y_unit="dB"), colorway=color.simplegradient(len(Z0),*color.default()[:2]),\
           legend="middle side" if len(Z0)>10 else "bottom right", ylabel="$\mathrm{S_{11}\:[dB]}$", **kwargs)



def beampatternRf(phi, theta, field_oc, theta_max=30, xrange=[0,30], yrange=[-30,0], **kwargs):
    '''
    Creates plot of beam pattern data. Exported from Zhang's or Shahab's PO tools.
    
    phi: matrix
        Matrix containing phi for each point in beam, shape=(num_phi, num_theta)
    theta: matrix
        Matrix containing theta for each point in beam, shape=(num_phi, num_theta)
    field_oc: matrix
        Field to plot amplitude in dB of, shape=(num_phi, num_theta)
    theta_max: number, kwarg
        passed on to x0 in plotly
    xrange: list, kwarg
        minimum and maximum for range x-axis
    yrange: list, kwarg
        minimum and maximum for range y-axis
    **kwargs:
        go to update_layout() in plotly()
    
    returns:
        go.Figure()
    '''
    theta *= 180/np.pi   # to degrees
    phi   *= 180/np.pi   # to degrees
    amp_oc = np.abs(field_oc)
    amp_oc = 20*np.log10(amp_oc/np.max(amp_oc)) # norm, to dB

    IDs = [f"\u03C6={round(p): >2}\u00B0, " for p in phi[:,0]]
    tickvals = func.adaptive_tickvals(xrange, 10, theta_max)

    plotly(theta, amp_oc, IDs=IDs, x0=theta_max, hlines=[-10],
           xrange=xrange, yrange=yrange, xaxis_tickmode='array', xaxis_tickvals=tickvals,
           xlabel=r'$\mathrm{\theta}\:\mathrm{[^\circ]}$', ylabel='$\mathrm{Normalized\:beam}\:\mathrm{[dB]}$',
           labels=dict(x="\u03B8", y="V/V\u2080", x_unit="\u00B0", y_unit="dB"), **kwargs)



def ndim_sweepCST(filename_fmt, mode, order, keys, labels, vals, savename=None, plot_S11=True, plot_Z11=False, Z0=55, ID_0=1, str_fmt=".3f", **layout):
    '''
    Function to ease plotting of S11, Re{Z11} and Im{Z11} of CST parameter sweeps.
    Should be easily adaptable by only changing what to plot using plotly()
    Refer to 'order' and 'keys' for correct input formatting.
    Works for any number of sweep parameters.
    
    filename_fmt: str
        String containing path to all files. Should contain '{:.0f}' for be 
        able to iterate over all run IDs. 
    mode: str
        Key for variable to keep constant in each plot
    order: list
        List of keys in order from outer to inner shell in the parameter sweep.
            Example using a=[1,2] with key "alpha"; b=[5,6] with key "beta"
                Run ID_0  : a=1; b=5
                Run ID_0+1: a=2; b=5
                Run ID_0+2: a=1; b=6
                Run ID_0+3: a=2; b=6
            Order should now be ["beta", "alpha"]
    keys: list
        List of keys corresponding to data in sweep. Does not have to match
        the order used in the sweep. This order of the sweep is defined by
        the argument 'order'. However, the keys used in keys should match
        the keys used in 'order.' 
            Using the example, we can use keys   = ["alpha", "beta"]
    labels: list
        List of labels corresponding to data in sweep. Index of each item
        should match the index of the corresponding key in keys.
            Using the example, we can use labels = ["\u03B1", "\u03B2"]
    vals: list
        List of value arrays corresponding to data in sweep. Index of each item
        should match the index of the corresponding key in keys.
            Using the example, we can use vals   = [a, b]
    savename: str or None, kwarg
        File name used for saving. Adds characters based on different plot prior to 
        the save type suffix (splits string at the period)
    plot_S11: bool, kwarg
        If True, also plots S11 rescaled to provided Z0
    plot_Z11: bool, kwarg
        If True, also plots Z11
    Z0: number, kwarg
        Adds Z0 to hlines for the Re{Z11} plot
        Rescales network to this impedance, only relevant if plot_S11 is True
    ID_0: number, kwarg
        First run ID for sweep
    str_fmt: str, kwarg
        Format of variables per run used for labels. Defaults to three decimal places
    layout: kwargs
        Passed on to sd.plotly() as kwargs
    
    returns:
        go.Figure for S11 (if plot_S11 is True), Re{Z11} and Im{Z11} (if plot_Z11 is True)
        Results in (2+plot_S11) * len(const_norm) plots
    '''
    import skrf as rf
    
    if not len(keys)==len(labels)==len(vals):
        raise Exception("Lengths of keys, labels, and vals should be the same.")
    if not bool(plot_S11+plot_Z11):
        raise Exception("No plots will be shown.")
    
    data = {key:[lab,val] for key,lab,val in zip(keys, labels, vals)}
    labels, vals = [k for k in zip(*[data[key] for key in order])] # Put data in right order

    idx_mode = order.index(mode)               # Index of the mode chosen
    not_mode = np.array(order) != mode         # Boolean array to select all but corresponding to mode
    const_norm = vals[idx_mode]

    n_i = np.array([len(val) for val in vals]) # Lengths of each variable
    n_total = np.prod(n_i)
    n_spacing = n_total//n_i[idx_mode]         # Number of IDs per constant

    # Building matrix from which we slice single run IDs,
    # formatted [#ID, x_1, x_2, ..., x_i-1, x_i+1, ..., x_len(vals)]
    IDs = np.zeros((len(vals)+1, *n_i))
    IDs[0]  = np.reshape(np.arange(n_total) + ID_0, n_i)  # First column #ID
    IDs[1:] = np.array(np.meshgrid(*vals, indexing='ij')) # Other columns x_i
    IDs = IDs[[True, *not_mode]]
    slices = lambda i: np.index_exp[:]*(idx_mode+1) + (i,) + np.index_exp[:]*(len(vals)-idx_mode-1)
    IDs_slice = lambda i: np.reshape(IDs[slices(i)], (len(vals), -1))

    # Other layout options:
    label_seq = ["{0: >3.0f}: "]+[label+"={"+str(i+1)+":0<3"+str_fmt+"}; " for i,label in enumerate(np.array(labels)[not_mode])]
    label_fmt = "".join(label_seq)
    layout["colorway"] = color.default()[:min(n_i[not_mode])]
    layout["legend_x"] = 1
    if isinstance(Z0, (float, int, complex)):
        h_re, h_im = [np.real(Z0)], [np.imag(Z0)]
    else:
        h_re, h_im = [], []
    
    title_fmt = labels[idx_mode] + "={:0<"+str_fmt+"}"
    titles = [title_fmt.format(const) for const in const_norm]
    plot_names = ["S11","Re(Z11)","Im(Z11)"]
    if savename is not None:
        prefix, suffix = savename.split(".")
        savename = [prefix+plot_name+"_"+title+"."+suffix for plot_name in plot_names for title in titles]
    else:
        savename = [None] * len(plot_names)*n_i[idx_mode]

    for i in range(n_i[idx_mode]):             # Loop over constants for mode
        layout["title_text"] = titles[i]
        name, S11, Z11_re, Z11_im = [], [], [], []

        for ID in zip(*IDs_slice(i)):          # Loop over runs within constant mode
            nw = rf.Network(filename_fmt.format(ID[0]))
            nw.renormalize(Z0)
            name.append(label_fmt.format(*ID))
            S11.append(nw.s_db.flatten())
            Z11_re.append(nw.z_re.flatten())
            Z11_im.append(nw.z_im.flatten())

        f = [nw.frequency.f_scaled]*n_spacing  # GHz
        if plot_S11:
            plotly(f, S11, name, labels=dict(x="f",x_unit="GHz",y="S11",y_unit="dB"), yrange=[None,0],
                   savename=savename[len(plot_names)*i], **layout)
        if plot_Z11:
            plotly(f, Z11_re, name, labels=dict(x="f",y="Re{Z11}",x_unit="GHz",y_unit="\u2126"), yrange=[0,None], hlines=h_re,
                   ylabel="$\mathrm{Re\{Z_{11}\}\:[\Omega]}$", savename=savename[len(plot_names)*i+1], **layout)
            plotly(f, Z11_im, name, labels=dict(x="f", y="Im{Z11}", x_unit="GHz", y_unit="\u2126"), hlines=h_im,
                   ylabel="$\mathrm{Im\{Z_{11}\}\:[\Omega]}$", savename=savename[len(plot_names)*i+2], **layout)
