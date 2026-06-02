from .plot import *

from . import lens
from . import antenna
from . import CPW
from . import KID
from . import TL
from . import func
from . import color

### Create templates
import plotly.graph_objects as go
import plotly.io as pio

colorbar_properties = dict(thickness=16, len=0.85, exponentformat="SI", title=dict(side="right", font_size=16))

pio.templates["base"] = go.layout.Template(
                           layout=go.Layout(dict(annotationdefaults=dict(arrowcolor="#929292", arrowhead=0, arrowwidth=1),
                                                 autotypenumbers='strict',
                                                 colorway=color.default(),
                                                 font_family='Space Mono',
                                                 hoverlabel=dict(align="left", font_family='Space Mono'),
                                                 hovermode='closest',
                                                 legend=dict(xanchor="left", x=0.01,
                                                             yanchor="top", y=0.99,
                                                             font_size=13),
                                                 margin=dict(l=70,r=70,t=60,b=80),
                                                 paper_bgcolor='rgba(0,0,0,0)',
                                                 shapedefaults=dict(line_color="#929292", label_textposition="middle center"),
                                                 title=dict(xanchor="center", x=0.5, xref="paper", subtitle_font_size=12),
                                                 xaxis=dict(automargin=True, exponentformat="SI", ticks="", title_font_size=24, 
                                                            title_standoff=15, type='linear'),
                                                 yaxis=dict(automargin=True, exponentformat="SI", ticks="", title_font_size=24, 
                                                            type='linear'),
                                                )),
                           data=dict(heatmap=[go.Heatmap(colorbar=colorbar_properties,
                                                         colorscale=color.toColorscale(), #color.mapscale(), "Cividis",
                                                         hoverongaps=False,
                                                         zsmooth=False)],
                                     contour=[go.Contour(colorbar=colorbar_properties,
                                                         colorscale=color.toColorscale())],
                                    ))

pio.templates["light"]= go.layout.Template(
                           layout=go.Layout(dict(font_color="#181932",
                                                 plot_bgcolor="#E5E6F3",
                                                 legend_bgcolor='white',
                                                 xaxis=dict(gridcolor='white',
                                                            linecolor='white',
                                                            zerolinecolor='white'),
                                                 yaxis=dict(gridcolor='white',
                                                            linecolor='white',
                                                            zerolinecolor='white'),
                                                )),
                           data=dict(heatmap=[go.Heatmap(colorbar=dict(outlinewidth=1, outlinecolor="black"))],
                                     contour=[go.Contour(colorbar=dict(outlinewidth=1, outlinecolor="black"))],
                                    ))

pio.templates["dark"] = go.layout.Template(
                           layout=go.Layout(dict(font_color='white',
                                                 plot_bgcolor="#181932",
                                                 legend_bgcolor='black',
                                                 xaxis=dict(gridcolor='black',
                                                            linecolor='black',
                                                            zerolinecolor='black'),
                                                 yaxis=dict(gridcolor='black',
                                                            linecolor='black',
                                                            zerolinecolor='black'),
                                                )),
                           data=dict(heatmap=[go.Heatmap(colorbar=dict(outlinewidth=1, outlinecolor="white"))],
                                     contour=[go.Contour(colorbar=dict(outlinewidth=1, outlinecolor="white"))],
                                    ))

pio.templates["xlog"] = go.layout.Template(
                           layout=go.Layout(dict(xaxis=dict(type='log', minor_showgrid=True, exponentformat="power"))
                                           ))

pio.templates["ylog"] = go.layout.Template(
                           layout=go.Layout(dict(yaxis=dict(type='log', minor_showgrid=True, exponentformat="power"))
                                           ))

pio.templates["contourmap"] = go.layout.Template(
                                 data=dict(contour=[go.Contour(contours=dict(coloring="heatmap", showlabels=True))]
                                          ))

# pio.template["largefont"] = go.layout.Template(
#                                layout=go.Layout(legend_font_size=20, title_subtitle_font_size=20,
#                                                 xaxis_title_font_size=36, yaxis_title_font_size=36),
#                                data=dict(heatmap=[colorbar_title_font_size=24], contour=[colorbar_title_font_size=24]),
#                                                )

# Make sure plots are shown when using a Jupyter Notebook:
try:
    from IPython import get_ipython
    if get_ipython().__class__.__name__ == "ZMQInteractiveShell":
        pio.renderers.default = "notebook_connected"
except Exception:
    pass
