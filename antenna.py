'''
This module contains a way to create an object of the class 'antenna.create' which allows the user to create an antenna object, which can for example be subsequently plotted. Also contains other functions to adapt the format of antenna dimensions.
'''

import numpy as np
import plotly.graph_objects as go
import warnings
from . import CPW, KID

class create:
    def __init__(self, L_ant, W_ant, W_1, W_2, d_ant, f0=70, c=3e8, tria=10):
        '''
        L_ant: float
            Antenna length in um
        W_ant: float
            Antenna width in um
        W_1: float
            Slot microstrip width in um
        W_2: float
            Width of strip connecting twin slots in um
        d_ant: float
            Distance between twin slots in um
        f0: number, kwarg
            Design (center) frequency in GHz
        c: number, kwarg
            Light speed in m/s
        tria: number, kwarg
            Dimension used while adding a sketch of the CPW to the figure, in um
            "height [in um] of triangles used to couple section sof differeht widths" - Jochem
        '''
        self.L_ant = L_ant
        self.W_ant = W_ant
        self.W_1 = W_1
        self.W_2 = W_2
        self.d_ant = d_ant
        self.dims = {"L_ant":L_ant, "W_ant":W_ant, "W_1":W_1, "W_2":W_2, "d_ant":d_ant}
        
        self.f0 = f0
        self.c = c
        self.lam = c/f0 *1e-3 # um
        self.tria = tria
        
    def draw(self, savename=None, xrange=[-0.14,0.23], yrange=[-0.185,0.185], size=800, TL=None, dec=0, substrate="Si", ground="NbTiN", color_sub="#757575", color_ground="#181932", color_hybrid="#D53F15", template='base+light', **kwargs):
        '''
        Create a schematic representation of the antenna.

        savename: string, kwarg
            file name for saving
        xrange: list, kwarg
            minimum and maximum fraction of c/f0 for x-axis
        yrange: list, kwarg
            minimum and maximum fraction of c/f0 for y-axis
        size: int, kwarg
            Width and height of plot
        TL: CPW.create, KID.hybrid or None, kwarg
            if an instance of the class CPW.create() is added, its dimensions
            are used to sketch a CPW in the figure.
            if an instance of the class KID.hybrid() is added, its .narrow and .wide
            dimensions are used to sketch the CPWs in the figure.
            Also adds name of the metal used for the central line to legend
        dec: number, kwarg
            Number of digits rounded to in legend for dimensions. Default is 0 (integer)
        substrate: str, kwarg
            Name of the substrate material, defaults to Si
        ground: str, kwarg
            Name of the substrate material, defaults to NbTiN. If a TL is given, it uses the 
            name of the ground material for the hybrid CPW
        color_sub: str, kwarg
            String containing color code for the substrate
        color_ground: str, kwarg
            String containing color code for the ground
        color_hybrid: str, kwarg
            String containing color code for the beta-tantalum CPW strip, only used if hybrid CPW is drawn
        template: str, kwarg
            string of go.layout.Template
        **kwargs:
            go to update_layout()

        returns:
            go.Figure()
        '''
        if round(xrange[1]-xrange[0],3) != round(yrange[1]-yrange[0],3):
            warnings.warn("Warning: xrange and yrange do not portray same range length; therefore, either x-axis or y-axis had to be stretched.") 
        
        # Dimensions
        lam   = self.lam  *1e-3 # mm

        L_ant = self.L_ant*1e-3 # mm
        W_ant = self.W_ant*1e-3 # mm
        W_1   = self.W_1  *1e-3 # mm
        W_2   = self.W_2  *1e-3 # mm
        d_ant = self.d_ant*1e-3 # mm

        xrange, yrange = np.array(xrange)*lam, np.array(yrange)*lam
        dx, dy, px = 0.01*lam, 0.01*lam, 7                                     # offsets for arrows and annotations
        offset   = dict(right=-px,center=0,left=px,top=-px,middle=0,bottom=px) # distances from arrows for annotations
        dl       = (xrange[1]-xrange[0])/25                                    # unit used for legend
        lW, lL   = 5*dl, 4*dl if TL is None else 6*dl                          # length and width of legend
        lx0, ly0 = xrange[1]-dx-lW, yrange[1]-dy-lL                            # anchor point for legends  
        fmt      = f"={{: >{5+(dec-1)*(dec>0)}.{dec}f}}\u03BCm"                # format used for values in legend

        # Setting anchor points and sizes for rectangles:
        points = [[[-W_ant/2,    -L_ant/2],
                   [-W_ant/2,     d_ant/2],
                   [ W_ant/2-W_1, d_ant/2],
                   [ W_ant/2-W_1,-L_ant/2]],
                  [[-W_ant/2,     d_ant/2],
                   [-W_ant/2,    -d_ant/2-W_1]],
                  [[-W_2/2,      -d_ant/2-W_1]]]
        rects  = [[ W_1,        (L_ant-d_ant)/2],
                  [ W_ant,       W_1],
                  [ W_2,         d_ant+2*W_1]]

        # Setting arrows and annotations:
        arrows = np.array([
                  [[-W_ant/2-dx,                 -L_ant/2   ], [-W_ant/2-dx,                  L_ant/2    ]], # L_ant
                  [[-W_ant/2,                    -L_ant/2-dy], [ W_ant/2,                    -L_ant/2-dy ]], # W_ant
                  [[ W_ant/2-W_1,                 L_ant/2+dy], [ W_ant/2,                     L_ant/2+dy ]], # W_1 1
                  [[ W_ant/2+dx,                  d_ant/2   ], [ W_ant/2+dx,                  d_ant/2+W_1]], # W_1 2
                  [[-W_2/2,      (TL is not None)*d_ant/2   ], [ W_2/2,      (TL is not None)*d_ant/2    ]], # W_2
                  [[min(-W_ant/2+W_1, -W_2/2-dx),-d_ant/2   ], [min(-W_ant/2+W_1, -W_2/2-dx), d_ant/2    ]]])# d_ant
        labels = ["L_{ant}", "W_{ant}", "W_1", "W_1", "W_2", "d_{ant}"] # labels for annotations
        annots  = np.mean(arrows, axis=1) # midpoints arrows for annotations
        anchors = [["right" ,"middle"], # L_ant
                   ["center","top"   ], # W_ant
                   ["center","bottom"], # W_1 1
                   ["left"  ,"middle"], # W_1 2
                   ["center","bottom"], # W_2
                   ["right" ,"middle"]] # d_ant
        
        fig = go.Figure()
        
        if isinstance(TL, KID.hybrid):
            drawCPW = True
            hybrid = TL.narrow
            ground = hybrid.mat_g.name
            l_hybrid = TL.l_hybrid*1e3 # mm
            
            # Draw connection narrow to wide CPW
            poly = np.array([[l_hybrid,                   -hybrid.W_tot/2*1e3],
                             [l_hybrid+self.tria  *1e-3, -TL.wide.W_tot/2*1e3],
                             [l_hybrid+self.tria  *1e-3, -TL.wide.S/2    *1e3], 
                             [l_hybrid+self.tria/2*1e-3, -(hybrid.S+TL.wide.S)/4*1e3]])
            poly = np.append(poly, poly[::-1]*np.array([1,-1]), axis=0)
            fig.add_shape(type="path", layer="between", line_color="rgba(0,0,0,0)", fillcolor=color_sub,
                          path=f"M {poly[0,0]} {poly[0,1]}" + "".join([f" L {x} {y}" for (x,y) in poly[1:]]) + " Z")
            
            # Add sizes+anchor points to rects+points for gaps of wide CPW
            points.append([[l_hybrid+self.tria*1e-3, TL.wide.S/2    *1e3],
                           [l_hybrid+self.tria*1e-3,-TL.wide.W_tot/2*1e3]])
            rects.append([xrange[1]-(l_hybrid+self.tria*1e-3), TL.wide.W*1e3])
        elif isinstance(TL, CPW.create):
            drawCPW = True
            hybrid = TL
            ground = hybrid.mat_g.name
            l_hybrid = xrange[1]
        else:
            drawCPW = False
        
        # Add legends
        fig.add_shape(x0=lx0,    y0=ly0,          x1=lx0+lW,    y1=ly0+lL,
                      type="rect", layer="between", line_color='white',      fillcolor='white')
        fig.add_shape(x0=lx0+dx, y0=ly0+dy,       x1=lx0+dx+dl, y1=ly0+dy+dl,
                      type="rect", layer="between", line_color=color_ground, fillcolor=color_ground)
        fig.add_shape(x0=lx0+dx, y0=ly0+lL-dy-dl, x1=lx0+dx+dl, y1=ly0-dy+lL,
                      type="rect", layer="between", line_color=color_sub,    fillcolor=color_sub)
        fig.add_annotation(x=lx0+dx+dl, y=ly0+dy+dl/2, xanchor='left', yanchor='middle', xshift=px,
                           text=ground, showarrow=False, font_color='black')
        fig.add_annotation(x=lx0+dx+dl, y=ly0-dy-dl/2+lL, xanchor='left', yanchor='middle', xshift=px,
                           text=substrate, showarrow=False, font_color='black')
        
        if drawCPW: # Add hybrid metal to legend and sizes+anchor points to rects+points for gaps of CPW
            fig.add_shape(x0=lx0+dx, y0=ly0+(lL-dl)/2, x1=lx0+dx+dl, y1=ly0+(lL+dl)/2,
                          type="rect", layer="between", line_color=color_hybrid, fillcolor=color_hybrid)
            fig.add_annotation(x=lx0+dx+dl, y=ly0+lL/2, xanchor='left', yanchor='middle', xshift=px,
                               text=hybrid.mat_c.name, showarrow=False, font_color='black')
            points.append([[0,  -hybrid.W_tot/2 * 1e3],
                           [0,   hybrid.S/2 * 1e3]])
            rects.append([l_hybrid, hybrid.W * 1e3])
        
        for h, label, var in zip(range(5), labels[:3:-1]+labels[2::-1], [L_ant, W_ant, W_1, W_2, d_ant][::-1]):
            for xshift, text in zip([-90, 0], ["$\large{\mathrm{"+label+"}}$", fmt.format(var*1e3)]):
                fig.add_annotation(x=lx0+lW, xshift=xshift-px, y=yrange[0]+dy, yshift=(h+1)*5*px, text=text, showarrow=False,
                                   xanchor='right', yanchor='middle', align="right", font=dict(color='white',size=16))

        # Plotting
        for r,ps in zip(rects,points): # Drawing rectangles at points
            for p in ps: 
                fig.add_shape(x0=p[0], y0=p[1], x1=r[0]+p[0], y1=r[1]+p[1],
                              type="rect", layer="between", line_color="rgba(0,0,0,0)", fillcolor=color_sub)
        
        if drawCPW: # Draw hybrid metal part of CPW
            fig.add_shape(x0=-(W_2+self.tria*1e-3)/2, y0=-hybrid.S/2 * 1e3, x1=l_hybrid+self.tria*1e-3, y1=hybrid.S/2 * 1e3,
                          type="rect", layer="between", line_color="rgba(0,0,0,0)", fillcolor=color_hybrid)

        for arrow, label, annot, anchor in zip(arrows, labels, annots, anchors): # Placing arrows and annotations
            marks = [arrow, arrow[::-1]] # arrows in both directions
            len_arrow = arrow[1]-arrow[0]
            if np.sum(len_arrow) < dl*1.2:   # displace arrows when space is too narrow
                marks += np.array([[-len_arrow], [len_arrow]])
            fig.add_annotation(x=annot[0], xanchor=anchor[0], xshift=offset[anchor[0]],
                               y=annot[1], yanchor=anchor[1], yshift=offset[anchor[1]],
                               text="$\Large{\mathrm{"+label+"}}$", 
                               showarrow=False, font_color='white')
            for mark in marks: # placing both arrows
                fig.add_trace(go.Scatter(x=mark[:,0], y=mark[:,1],
                                         marker=dict(symbol="arrow-up", angleref='previous',
                                                     color='white', size=10)))

        fig.update_layout(width=size, height=size, template=template, showlegend=False, plot_bgcolor=color_ground,
                          xaxis=dict(gridcolor='black', zerolinecolor='black',
                                     title="$\large{x\:\mathrm{[mm]}}$", range=xrange),
                          yaxis=dict(gridcolor='black', zerolinecolor='black',
                                     title="$\large{y\:\mathrm{[mm]}}$", range=yrange),
                          hovermode=False, **kwargs)
        fig.show()

        if savename is not None:
            fig.write_image(savename, width=size, height=size)

    def mask_input(self):
        '''
        Yields input to define the mask coordinates of the antenna in makeantennas.m
        '''
        points = [[           0,                              0],
                  [self.W_2  /2,                              0],
                  [self.W_2  /2,                   self.d_ant/2],
                  [self.W_ant/2,                   self.d_ant/2],
                  [self.W_ant/2,                   self.L_ant/2],
                  [self.W_ant/2-self.W_1,          self.L_ant/2],
                  [self.W_ant/2-self.W_1, self.W_1+self.d_ant/2],
                  [           0,          self.W_1+self.d_ant/2]]
        x, y = [str(s).strip("()") for s in list(zip(*points))]
        print("Copy and paste this for KID(1,:,:) [top right corner of antenna]:",
              f"[{x};...", 12*" "+f"{y}];", f"x_af = {self.W_2/2}", sep='\n')



def dims_to_physical(dims, f0=70, c=3e8):
    '''
    Converts dictionary from {alpha,beta,gamma,delta,epsilon} 
    to {L_ant,W_ant,W_1,W_2,d_ant} using:
        alpha   = L_ant/(c/f0)
        beta    = W_ant/L_ant
        gamma   = W_1  /L_ant
        delta   = W_2  /L_ant
        epsilon = d_ant/L_ant
    
    dims: dict
        dictionary containing factors for antenna in CST
    f0: number, kwarg
        Design (center) frequency in GHz
    c: number, kwarg
        Light speed in m/s
    '''
    lam = c/f0 *1e-3 # um
    L_ant = dims["alpha"]*lam
    return dict(L_ant = L_ant,
                W_ant = dims["beta"]*L_ant,
                W_1   = dims["gamma"]*L_ant,
                W_2   = dims["delta"]*L_ant,
                d_ant = dims["epsilon"]*L_ant)



def dims_to_factor(dims, f0=70, c=3e8):
    '''
    Converts dictionary from {L_ant,W_ant,W_1,W_2,d_ant}
    to {alpha,beta,gamma,delta,epsilon} using:
        alpha   = L_ant/(c/f0)
        beta    = W_ant/L_ant
        gamma   = W_1  /L_ant
        delta   = W_2  /L_ant
        epsilon = d_ant/L_ant
    
    dims: dict
        dictionary containing physical dimensions of antenna in um
    f0: number, kwarg
        Design (center) frequency in GHz
    c: number, kwarg
        Light speed in m/s
    '''
    lam = c/f0 *1e-3 # um
    return dict(alpha   = dims["L_ant"]/lam,
                beta    = dims["W_ant"]/dims["L_ant"],
                gamma   = dims["W_1"]/dims["L_ant"],
                delta   = dims["W_2"]/dims["L_ant"],
                epsilon = dims["d_ant"]/dims["L_ant"])



def writeCSV(TXTname, CSVname="Data/ID", xtitle = "f", ytitle = "S_11", L=1001, IDslice=slice(37,-8)):
    '''
    Function to convert txt exports from CST to csv-files.
    
    TXTname: str
        File to convert
    CSVname: str, kwarg
        Leading string to define csv-file names. Name will be: CSVname + str(ID) + '.csv'
    xtitle: str, kwarg
        x data column title in csv-file
    ytitle: str, kwarg
        y data column title in csv-file
    L: int, kwarg
        Number of data points per result in the export
    IDslice: slice, kwarg
        Slice defining how to extract ID from titles in txt export
        
    returns:
        list containing all ID strings
    '''
    import csv
    with open(TXTname, 'r') as TXTfile:
        stripped = np.reshape([l.strip() for l in TXTfile],(-1,L+3))
        IDs = [s[IDslice] for s in stripped[:,0]] # extract IDs
        Data = stripped[:,2:-1]                   # extract Data
        for i,s in enumerate(IDs):                # loop over different IDs (i.e., results)
            title = [xtitle, ytitle]
            subData = [title,*[l.split(16*' ') for l in Data[i]]]
            with open(CSVname+s+".csv", 'w') as CSVfile:
                writer = csv.writer(CSVfile)
                writer.writerows(subData)         # write csv-file
    return IDs