'''
This module contains several functions to ease plotting using Plotly. plotly() contains standard methods to plot x- and y-data.
'''

import numpy as np
import plotly.graph_objects as go
from . import func, color
from .KID import resonator

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



def KID_plots(f_design, networks, s=[0,1], return_pvar=False, plot_CPW=False, savefolder=None, alpha_guess=-0.1, Qc_guess=3e3, Qi_guess=3e3):
    '''
    Function to plot (and save), curve-fit, and retrieve resonator parameters for KIDs. 
    KID data is provided to KID_plots() through f_design and networks, the latter containing rf.Network instances
    (from skrf) with .s2p files as inputs (two-port simulation data). Curve-fits are performed using scipy.optimize.curve_fit
    
    f_design: array
        Design resonance frequencies, in GHz
    networks: list
        List of rf.Network networks for each of the design resonance frequencies. Assumes formatting for 
        file names to contain at least one underscore ("_") for savename formatting
    s: list, kwarg
        Indices to plot correct s-parameter data. Should normally not be changed.
    return_pvar: bool, kwarg
        If True, fitted variable values pvars and respective standard deviations pstds are returned,
        formatted as pvars[i] = [alpha, f0, Qc, Qi] with i in range(len(f_design))
        If False, pvars and pstds are printed alongside the plots
    plot_CPW: bool, kwarg
        If True, s-parameter data before removing the phase shift added due to the CPW is plotted
        alongside the other plots
    savefolder: str, kwarg
        If not None, the plots are saved in this folder
    alpha_guess: number, kwarg
        Phase coefficient guess provided to p0 in curve_fit. Alpha is processed through exp(1j*alpha*f)
    Qc_guess: number, kwarg
        Coupling quality factor guess provided to p0 in curve_fit
    Qi_guess: number, kwarg
        Internal quality factor guess provided to p0 in curve_fit
    '''
    from scipy.optimize import curve_fit
    
    keys = ["f", "mag2", "rad_CPW", "re_CPW", "im_CPW", "rad", "re", "im", "IDs", "lkwargs"]
    data = {key:[] for key in keys}
    pvars, pstds = [], []
    
    S12 = lambda f, alpha, f0, Qc, Qi: resonator(f, f0, Qc, Qi) * np.exp(1j*alpha*f)
    def S12_fitfunc(f, alpha, f0, Qc, Qi): # f and f0 in GHz
        complex_vals = S12(f, alpha, f0, Qc, Qi)
        return np.concatenate([np.real(complex_vals), np.imag(complex_vals)])
    
    for f_ID, nw in zip(f_design, networks):        
        f_i = nw.frequency.f_scaled
        s_mag_i, s_re_i, s_im_i = nw.s_mag[:,s[0],s[1]], nw.s_re[:,s[0],s[1]], nw.s_im[:,s[0],s[1]]
        
        pvar, pcov = curve_fit(S12_fitfunc, f_i, np.concatenate([s_re_i, s_im_i]),
                               p0    = (alpha_guess, f_i[np.argmin(s_mag_i)], Qc_guess, Qi_guess),
                               bounds=([    -np.inf,             np.min(f_i),        0,        0],
                                       [     np.inf,             np.max(f_i),   np.inf,   np.inf]))
        
        [alpha, f0, Qc, Qi] = pvar
        [_, _, std_Qc, std_Qi] = np.sqrt(np.diag(pcov))
        pvars.append(pvar)
        pstds.append(np.sqrt(np.diag(pcov)))
        
        if not return_pvar:
            print(f"Design {f_ID:.2f}GHz:", f"f0 = {f0:.3f}GHz",
                  f"Qc = {int(Qc)} \u00B1 {std_Qc:.1g}", f"Qi = {int(Qi)} \u00B1 {std_Qi:.1g}\n", sep="\n")
        
        res = nw.s[:,s[0],s[1]] * np.exp(-1j*alpha*f_i)
        res_fit = resonator(f_i, f0, Qc, Qi)
        S12_fit = S12(f_i, alpha, f0, Qc, Qi)

        data["f"]       += [f_i]*2
        data["mag2"]    += [s_mag_i**2,            np.abs(S12_fit)**2]
        data["rad_CPW"] += [nw.s_rad[:,s[0],s[1]], np.angle(S12_fit)]
        data["re_CPW"]  += [s_re_i,                np.real(S12_fit)]
        data["im_CPW"]  += [s_im_i,                np.imag(S12_fit)]
        data["rad"]     += [np.angle(res),         np.angle(res_fit)]
        data["re"]      += [np.real(res),          np.real(res_fit)]
        data["im"]      += [np.imag(res),          np.imag(res_fit)]
        data["IDs"]     += [f"Design f0 = {f_ID:.2f}GHz", f"Fit: Qi={int(Qi)}; Qc={int(Qc)}"]
        data["lkwargs"] += [{},{"line_dash":"dash","line_width":1}]
    
    if savefolder is not None:
        sizes = [(730,350), (500,300), (350,350)]
    else:
        sizes = [(730,350), (500,350), (500,500)]
    colorway = sum([[c, "black"] for c in color.default()[:len(f_design)]], [])
    
    # Plotting all quantities:
    def get_savename(suffix):
        if savefolder is not None:
            return savefolder + "/" + nw.name.rsplit(sep="_", maxsplit=1)[0] + suffix + ".svg"
        else:
            return None
    
    plotly(data["f"], data["mag2"], data["IDs"], linekwargs=data["lkwargs"], colorway=colorway, size=sizes[0],
           yrange=[None,1], vlines=f_design, legend_x=1.22, title_text=f"Qc = {Qc_guess:.1g}", 
           labels=dict(x="",x_unit="",y=f"|S_{{{s[0]+1}{s[1]+1}}}|^2",y_unit=""),
           savename=get_savename("_s_mag"))
    if plot_CPW:
        plotly(data["f"], data["rad_CPW"], linekwargs=data["lkwargs"], colorway=colorway, size=sizes[1],
               labels=dict(x="f",x_unit="GHz",y=f"\u2220S_{{{s[0]+1}{s[1]+1}}}",y_unit="rad"),
               savename=get_savename("_s_rad_CPW"))
        plotly(data["re_CPW"], data["im_CPW"], linekwargs=data["lkwargs"], colorway=colorway, 
               xrange=[0,1], yrange=[-0.5,0.5], hlines=[0], vlines=[0], size=sizes[2],
               labels=dict(x="Re\{S_"+f"{{{s[0]+1}{s[1]+1}}}"+"\}",x_unit="",
                           y="Im\{S_"+f"{{{s[0]+1}{s[1]+1}}}"+"\}",y_unit=""),
               savename=get_savename("_s_re_im_CPW"))
    plotly(data["f"], data["rad"], linekwargs=data["lkwargs"], colorway=colorway, size=sizes[1],
           labels=dict(x="f",x_unit="GHz",y=f"\u2220S_{{{s[0]+1}{s[1]+1}}}",y_unit="rad"),
           savename=get_savename("_s_rad"))
    plotly(data["re"], data["im"], linekwargs=data["lkwargs"], colorway=colorway, size=sizes[2],
           xrange=[0,1], yrange=[-0.5,0.5], hlines=[0], vlines=[0],
           labels=dict(x="Re\{S_"+f"{{{s[0]+1}{s[1]+1}}}"+"\}",x_unit="",
                       y="Im\{S_"+f"{{{s[0]+1}{s[1]+1}}}"+"\}",y_unit=""),
           savename=get_savename("_s_re_im"))
    
    if return_pvar:
        return pvars, pstds



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
            nw.renormalize(np.conjugate(Z0))
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



def sweepCST(filename_prefix, l_sweep, dims, step, BP_f, BP_ratio, BP_fvals=None, mode="L_ant+W_ant+W_1+W_2+d_ant", f0=70, dl=0, frange=[50,90], xrange=[60,80], showBW=False, Lk=1, size=(600,500), template='base+light', savename=None, colors_sweep=color.default(), colorList=color.default()):
    '''
    Function to plot sweeps in all of the five variables used for antenna scaling.
    CST setup:
        1. Clear any previous results, such that first simulation will have ID=1
        2. Have one base set of variables [L_ant, W_ant, W_1, W_2, d_ant]
        3. Set up a series of sweeps where variables can be swept while keeping 
           the others constant. It therefore assumes that the base set is the 
           center of each of the sweeps and will thus by simulated only once by CST. E.g.,
           Sweep1: varying L_ant, other variables constant (base set is simulated)
           Sweep2: varying W_ant, other variables constant (base set is not simulated again)
           etc.
        4. Perform linear sweeps with step size 'step' and length 'l_sweep'. E.g.,
           L_ant = [96, 98, 100, 102, 104] for base length L_ant=100, step=2, and l_sweep=5.
        5. Export the results as Touchstone files (for these antennas, we expect s1p-files).
           For sweeps in all five variables, this would yield 
           N = (5 variables) X l_sweep - 4 different files (we subtract 4, as CST
           won't simulate any duplicate sets of dimensions twice) with IDs = {1,2,...,N}
    
    filename_prefix: str
        string containing file location and prefix for s1p-files from CST.
        Example: 'datafolder/TwinSlot'. Adds IDs and ".s1p" to find files.
    l_sweep: int
        odd number to define length of sweep in each of the five variables.
    dims: dict
        dictionary containing dimensions of antenna, using same inputs as draw(),
        using the dict keys: [L_ant, W_ant, W_1, W_2, d_ant]
    step: number
        if a sweep in size, the difference in size from base set of variables defined through dims in um.
        otherwise, defines step size in swept quantity
    BP_f: array
        array containing frequencies at which BP-filter is known
    BP_ratio: array
        array containing transmission of BP-filter
    BP_fvals: list, kwarg
        list of frequencies defining bandwidth. If None, xrange is used instead.
    mode: str, kwarg
        string containing names of sweeps to plot, divided by '+', in the order they were
        simulated in CST.
    f0: number, kwarg
        Design (center) frequency in GHz
    dl: int, kwarg
        plot 2*dl less networks. Trims sweeps on boundaries by dl.
        Example: when dl=1, the sweep L_ant = [96, 98, 100, 102, 104] will only
        have L_ant = [98, 100, 102] plotted.
    frange: list, kwarg
        lower and upper limit of frequency used from CST data. I.e., trims any network
        through network[f'{frange[0]}-{frange[1]}ghz']
    xrange: list, kwarg
        minimum and maximum for range x-axes
    showBW: boolean, kwarg
        passed through to plotly(). If True, this function will also return a dictionary
        containing data from func.findBW()
    Lk: number, kwarg
        Kinetic inductance in pH. Only used when sweep in Lk is performed and considered as median
    size: tuple, kwarg
        tuple containing (width,height) for size of figure
    template: str, kwarg
        string of go.layout.Template
    savename: string, kwarg
        file name used for saving
    colors_sweep: list, kwarg
        passed through to plotly(). List containing color code strings, should ideally have length=l_sweep
    colorList: list, kwarg
        list containing color code strings for each of the variables in plots, should ideally have at least length=5
        
    returns:
        5 go.Figure() from plotly() for each of the variables
        2 go.Figure() for tolerance plots for S11(f=f0) and BW
    '''
    from plotly.subplots import make_subplots
    import skrf as rf
    
    l_sweep_half = l_sweep//2
    names  = mode.split("+")
    if l_sweep_half == l_sweep/2:
        raise Exception("Only works for sweeps of odd length.")
    if len(BP_f) != len(BP_ratio):
        raise Exception("Lengths of BP_f and BP_ratio do not match.")
    if not np.all(np.diff(BP_f)>0):
        raise Exception("Frequency must be in ascending order and without duplicates.")
    if dl > l_sweep_half:
        raise Exception("dl can't be set larger than half of the sweep length.")
    if not np.any([np.all([name in dims.keys() for name in names]), mode == "etch", mode == "all", mode == "Lk"]):
        raise Exception("Error in key(s) used in mode. Try a combination of 'L_ant', 'W_ant', 'W_1', 'W_2', and 'd_ant', or use 'etch', 'all' or 'Lk'.")
    
    # set up networks to plot:
    indices= [list(np.arange(dl,l_sweep-dl)),
              *[[*np.arange(n*(l_sweep-1)+1+dl, n*(l_sweep-1)+l_sweep_half+1),\
                 l_sweep_half,                                                \
                 *np.arange(n*(l_sweep-1)+l_sweep_half+1, (n+1)*l_sweep-n-dl)]\
                for n in range(1,len(names))]
             ]
    diff_qt = np.arange(-(l_sweep_half-dl)*step, (l_sweep_half-dl+1)*step, step) # difference in quantity from ID=l_sweep_half+1
    
    if mode == "all":
        size_qt = np.array([[f"{d: }" for d in diff_qt]]) # physical difference in sizes in um
        names = ["Vary all"]
        xlabel = '$\mathrm{\Delta}\:\mathrm{[\mu m]}$'
        titles = ["\Delta"]
        IDs = np.arange(l_sweep)+1
    elif mode == "etch":
        size_qt = np.array([[f"{d: }" for d in diff_qt]]) # physical difference in sizes in um
        names = ["Under-<br>/overetch"]
        xlabel = '$\mathrm{\Delta}\:\mathrm{[\mu m]}$'
        titles = [r"\Delta \:\text{(under-/overetch of NbTiN edges)}"]
        IDs = np.arange(l_sweep)+1
    elif mode == "Lk":
        diff_qt += Lk
        size_qt = np.array([diff_qt], dtype=str) # different Lk in pH
        xlabel = '$\mathrm{L_k}\:\mathrm{[pH]}$'
        names = ["Vary Lk"]
        titles = ["L_k"]
        IDs = np.arange(l_sweep)+1
    else:
        size_qt = np.array([diff_qt + d for d in [dims[key] for key in names]], dtype=str) # physical sizes in um
        xlabel = '$\mathrm{\Delta}\:\mathrm{[\mu m]}$'
        titles = [name[:2]+'{'+name[2:]+'}' for name in names]
        IDs = np.arange(l_sweep*5-4)+1
    
    # create networks
    S11, S11_lin, frequency = [], [], []
    for ID in IDs:
        nw = rf.Network(filename_prefix+str(ID)+".s1p")
        nw = nw[f'{frange[0]}-{frange[1]}ghz']

        frequency.append(nw.frequency.f_scaled)
        S11.append(nw.s_db.flatten())
        S11_lin.append(nw.s_mag.flatten())
    
    # set up BP-filter, using that frequency settings match for all networks in sweep:
    BP_ratio = np.interp(nw.frequency.f_scaled, BP_f, BP_ratio)
    BP_fvals = xrange if BP_fvals is None else BP_fvals
    BP_ratio_sum = np.sum(BP_ratio, where=(nw.frequency.f_scaled>BP_fvals[0])*(nw.frequency.f_scaled<BP_fvals[1]))
    df = np.arange(1-len(BP_ratio),len(BP_ratio))*nw.frequency.step_scaled
    
    # plotting different figures:
    colors_sweep = colors_sweep[:l_sweep_half-dl] + [colors_sweep[l_sweep_half]] + colors_sweep[l_sweep_half+1+dl:]
    figS11, figBW = go.Figure(), go.Figure() # S11(f=f0) plot, BW plot
    figCC = make_subplots(rows=2, shared_xaxes=True, row_heights=[1, 0.5], vertical_spacing=0.07,\
                          cols=2, shared_yaxes=True, column_widths=[0.85, 0.15], horizontal_spacing=0.02,\
                          subplot_titles=("Antenna S11 bandwidth",\
                                          "BP-filter",\
                                          f"Cross-correlation 1-S11 and BP-filter at f={f0}GHz")) # cross-correlation plot
    savename = savename.split(".") if savename is not None else None
    for i, name in enumerate(names):
        xdata = np.array([frequency[index] for index in indices[i]])
        ydata = np.array([S11[index] for index in indices[i]])
        ydata_lin = np.array([S11_lin[index] for index in indices[i]])

        # scatter plot of sweep:
        plotly(xdata, ydata, IDs=[s+'\u03BCm: ' for s in size_qt[i]], x0=f0, showBW=showBW, size=size, xrange=xrange, yrange=[None,0],\
               legend=dict(x=1.01, y=0.5, yanchor='middle'), labels=dict(x="f", y="S11", x_unit="GHz", y_unit="dB"),\
               title=dict(text=r"$\text{Sweep in }%s$"%titles[i],xanchor='center',xref='paper',x=0.5,yanchor='top',y=0.92),\
               colorway=colors_sweep, template=template, savename=savename[0]+f"-{name.lower().replace('<br>/','')}"+"."+savename[1] if savename is not None and savename[1]!="html" else None)

        # find S11(f=f0):
        args = np.argsort(np.abs(xdata[0]-f0))
        interArg = args[:2]
        y0 = func.inter(ydata[:,interArg[0]], xdata[:,interArg[0]], ydata[:,interArg[1]], xdata[:,interArg[1]], f0)
        figS11.add_trace(go.Scatter(x=diff_qt, y=y0, name=name, line_color=colorList[i]))

        # find bandwidth BW, defined as distance between crossings with -10dB line:
        extremes = np.argmin(ydata, axis=-1) # arguments of extreme value
        BWs, BW_fvals = zip(*[func.findBW(x, y, extreme, y_BW=-10) for x, y, extreme in zip(xdata, ydata, extremes)])
        figBW.add_trace(go.Scatter(x=diff_qt, y=BWs, name=name, line_color=colorList[i]))
        
        # find cross correlation between transmission and BP-filter:
        cc_visible = True if not bool(i) or savename is not None else 'legendonly'
        
        figCC.add_trace(go.Scatter(x=diff_qt, y=np.array(BW_fvals)[:,0], name=name, legendgroup=name, visible=cc_visible,                                   fill=None, mode='lines', line_color=colorList[i], line_dash='dash', showlegend=False))
        figCC.add_trace(go.Scatter(x=diff_qt, y=np.array(BW_fvals)[:,1], name=name, legendgroup=name, visible=cc_visible,                                   fill='tonexty', mode='lines', line_color=colorList[i], line_dash='dash', showlegend=False))
        eta_cc = np.sum(BP_ratio*(1-ydata_lin), axis=1) / BP_ratio_sum
        figCC.add_trace(go.Scatter(x=diff_qt, y=eta_cc, name=name, legendgroup=name, visible=cc_visible,\
                                   line_color=colorList[i], showlegend=True), row=2, col=1)
    
    figCC.add_trace(go.Scatter(x=10*np.log10(BP_ratio), y=nw.frequency.f_scaled, line_color='white', showlegend=False),\
                    row=1, col=2)
    figCC.add_shape(x0=-30, y0=frange[0]-50, x1=5, y1=BP_fvals[0], type="rect", line_dash='dash',\
                    line_color='rgb(255,255,255)', fillcolor='rgba(255,255,255,0.3)', row=1, col=2)
    figCC.add_shape(x0=-30, y0=BP_fvals[1], x1=5, y1=frange[1]+50, type="rect", line_dash='dash',\
                    line_color='rgb(255,255,255)', fillcolor='rgba(255,255,255,0.3)', row=1, col=2)
    
    # Setting layout + showing and saving plots:
    figS11.update_yaxes(title='$\mathrm{S_{11}(f=f_0)}\:\mathrm{[dB]}$')
    figBW.update_yaxes(title='$\mathrm{BW}\:\mathrm{[GHz]}$')
    figCC.update_yaxes(title='$\mathrm{f}\:\mathrm{[GHz]}$', range=xrange, row=1, col=1)
    figCC.update_yaxes(title='$\mathrm{\eta_{CC}}\:\mathrm{[-]}$', row=2, col=1)
    figCC.update_xaxes(title=r'$\text{Ratio}\:\mathrm{[dB]}$', title_standoff=0, range=[-10,0],\
                       showticklabels=True, tickmode = 'linear', tick0=-30, dtick=5, row=1, col=2)
    ticks_xaxes = dict(range=[diff_qt[0],diff_qt[-1]],\
                       tickmode = 'linear', tick0 = diff_qt[0], dtick = step)
    
    for fig, fig_name, nrows in zip([figCC, figBW, figS11],["CC","BW","S11"], [1.5, 1, 1]):
        fig.update_layout(width=size[0], height=size[1]*nrows, template=template, showlegend=True)
        
        if fig_name == "CC":
            fig.update_layout(legend=dict(xanchor="left", x=0.86, yanchor="middle", y=0.155))
            fig.update_xaxes(**ticks_xaxes, row=1, col=1)
            fig.update_xaxes(title=xlabel, **ticks_xaxes, row=2, col=1)
            fig.update_xaxes(tickmode='array', tickvals=[-10,-3,0], row=1, col=2)
        else:
            fig.update_xaxes(title=xlabel, **ticks_xaxes)
        
        if savename is not None:
            if savename[1]=="html":
                savefunction = fig.write_html
                js = '''document.body.style.backgroundColor = "#000"; '''
                savekwargs = dict(include_plotlyjs="cdn", include_mathjax='cdn', post_script=[js])
            else:
                savefunction = fig.write_image
                savekwargs = dict(width=size[0], height=size[1]*nrows)
            savefunction(savename[0]+fig_name+"."+savename[1], **savekwargs)
        fig.show()