'''
This module contains a way to create an object of the class 'lens.create' which allows the user to create a lens object, which can for example be subsequently plotted.
'''

import numpy as np
import plotly.graph_objects as go
from . import color

get_R_sphere = lambda D, theta, ext: D/2 * ( np.sqrt( 1 - (ext*np.sin(theta))**2 ) - ext*np.cos(theta) ) / (1-ext**2) / np.sin(theta)
get_ext      = lambda H, R_FO, theta0: (H**2 - R_FO**2) / (H**2 + R_FO**2 - 2*H*R_FO*np.cos(theta0))
get_D        = lambda R_sphere, theta, ext: 2*R_sphere*(1-ext**2)*np.sin(theta) / (np.sqrt(1 - (ext*np.sin(theta))**2) - ext*np.cos(theta))

class create():
    def __init__(self, D, theta0, ext):
        '''
        D: float
            Diameter lens
        theta0: float
            Design opening angle hemispherical lens
        ext: float, kwarg
            Extension in terms of R_sphere
        '''
        self.D = D
        self.theta0_deg = theta0
        self.theta0 = self.theta0_deg * np.pi/180
        self.ext = ext
        
        self.R_sphere = get_R_sphere(self.D, self.theta0, self.ext) # radius hemispherical lens
        self.H = (1+ext) * self.R_sphere # total height
        
        self.h = D/2 / np.tan(self.theta0) # height of substrate without hemispherical lens
        self.R_FO = D/2 / np.sin(self.theta0) # radius FO sphere
        self.f_num = self.R_FO/D # f-number
    
    def theta_max(self, D_ap, Z_tot):
        '''
        Derives the maximum opening angle given the setup the lens is placed in.
        
        D_ap: float
            Diameter aperture
        Z_tot: float
            Height from aperture to bottom of lens

        Returns: 
            Maximum theta (opening angle) in degrees
        '''
        Z_ap = Z_tot - self.H
        return np.arctan(D_ap /2 / Z_ap) *180/np.pi # degrees
    
    def mesh_D(self, theta, t_wafer):
        '''
        Derives the mesh diameter given opening angle and thickness of the wafer on which it is placed on the backside.
        
        theta: float
            Opening angle in rad
        t_wafer: float
            Thickness of wafer
            
        returns:
            Diameter of mesh on the backside of the wafer
        '''
        return 2*t_wafer*np.tan(theta)
    
    def update_ext(self, H_ref):
        '''
        Updates extension (and R_sphere, H) given known lens dimensions and a reference H (height from focal point to top of lens).
        This keeps R_FO constant.
        
        H_ref: float
            Height from focal point to top of lens
        '''
        self.H = H_ref
        self.ext = get_ext(self.H, self.R_FO, self.theta0) # ext dependent on H
        self.R_sphere = get_R_sphere(self.D, self.theta0, self.ext) # R_sphere dependent on ext
    
    def update_theta(self, H_ref):
        '''
        Updates theta0 (and R_FO, h, H, ext, f_num) given known lens dimensions and a reference H (height from focal point to top of lens).
        This keeps R_sphere constant.
        
        H_ref: float
            Height from focal point to top of lens
        '''
        self.h -= self.H - H_ref
        self.H = H_ref
        self.theta0 = np.arctan(self.D/2 / self.h)
        self.theta0_deg = self.theta0 * 180/np.pi
        self.R_FO = self.D/2 / np.sin(self.theta0) # R_FO dependent on theta
        self.ext = get_ext(self.H, self.R_FO, self.theta0) # ext dependent on H and theta
        self.f_num = 1/2/np.sin(self.theta0) # f_num dependent on theta
    
    def update_D(self, H_ref):
        '''
        Updates D (and R_sphere, H, R_FO, h) given known lens dimensions and a reference H (height from focal point to top of lens).
        This keeps theta0 and ext constant. 
        
        H_ref: float
            Height from focal point to top of lens
        '''
        self.H = H_ref
        self.R_sphere = self.H / (1+self.ext)
        self.D = get_D(self.R_sphere, self.theta0, self.ext)
        self.h = self.D/2 / np.tan(self.theta0) # h dependent on D
        self.R_FO = self.D/2 / np.sin(self.theta0) # R_FO dependent on D
    
    def draw(self, t_wafer=0.350, unit="mm", color_Si="#c0c0c0", size=500, name="Ext. hemispherical lens", savename=None, template='base+light'):
        '''
        Create a schematic representation of the lens
        
        t_wafer: number, kwarg
            Thickness of wafer
        unit: str, kwarg
            Unit of length used for all variables
        color_Si: str, kwarg
            String containing color code for the silicon
        size: number, kwarg
            height of plot; width is calculated as 5/4*height
        name: str, kwarg
            name of the lens
        savename: string, kwarg
            file name for saving
        template: str, kwarg
            string of go.layout.Template
            
        returns:
            go.Figure()
        '''
        fig = go.Figure()
        colors = color.default()
        
        for z in [t_wafer, self.h, self.H]:
            fig.add_hline(z, line_color='rgba(0,0,0,0.3)', line_dash='dash',
                          annotation_text=f"{z:.3f}", annotation_position='right')
        for r in [-self.D/2, self.D/2]:
            fig.add_vline(r, line_color='rgba(0,0,0,0.3)', line_dash='dash',
                          annotation_text=f"{r:.3f}", annotation_position='top')
        
        # R_sphere and R_FO
        for R, y0, c in zip([self.R_sphere, self.R_FO],
                            [(self.ext-1)*self.R_sphere, -self.R_FO],
                            [colors[0], colors[1], colors[-1]]):
            fig.add_shape(type="circle",
                          x0=-R, y0=y0, 
                          x1=R, y1=y0+2*R,
                          line_color=color_Si, fillcolor=color_Si,
                          layer='between')
            fig.add_shape(type="circle",
                          x0=-R, y0=y0, 
                          x1=R, y1=y0+2*R,
                          line_color=c, layer='between')        
        
        # R_sphere
        fig.add_trace(go.Scatter(x=[0,0], y=[self.ext*self.R_sphere, (self.ext+1)*self.R_sphere],
                                 line_color=colors[0], mode="lines", showlegend=False))
        fig.add_trace(go.Scatter(x=[0,-self.D/2], y=[self.ext*self.R_sphere, self.h],
                                 line_color=colors[0], mode="lines",
                                 name='R_sphere = %.3f%s'%(self.R_sphere, unit)))
        fig.add_annotation(x=-self.D/4, y=(self.ext*self.R_sphere+self.h)/2, 
                           xanchor='center', yanchor='bottom',
                           text='$\large{\mathrm{R_{sphere}}}$', 
                           textangle=np.arccos(self.D/2/self.R_sphere)*180/np.pi, 
                           font=dict(color=colors[0], size=15), showarrow=False)
        
        # R_FO
        fig.add_trace(go.Scatter(x=[0,-self.D/2], y=[0, self.h],
                                 line_color=colors[1], mode="lines",
                                 name='R_FO     = %.3f%s'%(self.R_FO, unit)))
        fig.add_annotation(x=-self.D/4, y=self.h/2, 
                           xanchor='center', yanchor='bottom', showarrow=False,
                           text='$\large{\mathrm{R_{FO}}}$', textangle=90-self.theta0_deg, 
                           font=dict(color=colors[1], size=15))
        
        # ext
        fig.add_trace(go.Scatter(x=[0,0], y=[0, self.ext*self.R_sphere], mode="lines",
                                 line_color=colors[-1], line_dash='dash',
                                 name=f'Ext. = {self.ext:.3f}\u2022R_sphere'))
        fig.add_annotation(x=0, y=self.ext/2 * self.R_sphere, 
                           xanchor='left', yanchor='middle', align='left', showarrow=False,
                           text='$\mathrm{\:Extension}$', font=dict(color=colors[-1], size=15))
        
        # Silicon layer
        fig.add_shape(type="rect",
                      x0=-self.D, y0=0,
                      x1=self.D, y1=self.h,
                      line_color=color_Si, fillcolor=color_Si,
                      layer='between')
        
        # theta_0
        R_theta = self.ext/2*self.R_sphere
        fig.add_shape(type="circle", x0=-R_theta, y0=-R_theta, x1=R_theta, y1=R_theta,
                      line_color=colors[-1], layer='between')
        fig.add_shape(type="rect", x0=0, y0=0, x1=R_theta, y1=R_theta,
                      line_color=color_Si, fillcolor=color_Si, layer='between')
        fig.add_shape(type="path", path=f" M 0 0 L {-R_theta} 0 L {-R_theta} {R_theta/np.tan(self.theta0)} Z",
                      line_color=color_Si, fillcolor=color_Si, layer='between')
        fig.add_annotation(x=-R_theta*np.cos((np.pi-self.theta0)/2), y=R_theta*np.sin((np.pi-self.theta0)/2),
                           xanchor='center', yanchor='bottom', showarrow=False,
                           text='$\large{\mathrm{\u03B8_{0}}}$', textangle=-self.theta0_deg/2, 
                           font=dict(color=colors[-1], size=15))
        
        # Nothing below z=0
        fig.add_shape(type="rect",
                      x0=-self.D, y0=-self.D,
                      x1=self.D, y1=0,
                      line_color='rgba(255,255,255,0)', fillcolor='rgb(255,255,255)',
                      layer='between')
        
        fig.add_annotation(x=self.D/2, y=self.h, xanchor='left', yanchor='bottom',
                           text="Air", showarrow=False, font_size=13)
        fig.add_annotation(x=self.D/2, y=self.h, xanchor='left', yanchor='top',
                           text="Si", showarrow=False, font=dict(color=colors[-1], size=13))
        if t_wafer:
            fig.add_annotation(x=self.D/2, y=t_wafer, xanchor='left', yanchor='top', yshift=5,
                               text="Wafer", showarrow=False, font=dict(color=colors[-1], size=13))

        fig.update_xaxes(title="$\large{r\:\mathrm{[%s]}}$"%unit, range=[-self.D*5/8,self.D*5/8])
        fig.update_yaxes(title="$\large{z\:\mathrm{[%s]}}$"%unit, range=[0,self.D], scaleanchor="x", scaleratio=1)
        fig.update_layout(title=dict(text=f"{name}<br>D = {self.D:.3f}{unit}; \u03B8\u2080 = {self.theta0_deg:.1f}\u00B0",
                                     xanchor="center", x=0.5, yanchor="bottom"),
                          width=5/4*size, height=size, template=template, showlegend=True, hovermode=False)

        if savename is not None:
            fig.write_image(savename, width=5/4*size, height=size)

        fig.show()