import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

def set_default_fig_layout(fig, xtickvals, xticktext, ytickvals, yticktext):
    fig.add_annotation(
        text="CC BY Epoch",
        xref="paper",
        yref="paper",
        x=1.0,
        y=-0.14,
        showarrow=False,
        font=dict(
            size=12,
            color="#999999"
        ),
    )
    fig.update_layout(
        xaxis = dict(
            tickmode='array',
            tickvals=xtickvals,
            ticktext=xticktext,
        ),
        yaxis=dict(
                tickmode='array',
                tickvals=ytickvals,
                ticktext=yticktext,
        )
    )
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="right",
        x=1
    ))
    fig.update_layout(
        autosize=False,
        width=800,
        height=600,
        title_x=0.5,
        margin=dict(l=100, r=30, t=80, b=80),
    )
    return fig


def save_plot(fig, folder, filename, extensions=['png', 'svg', 'pdf'], scale=2):
    for ext in extensions:
        fig.write_image(folder + filename + '.' + ext, scale=scale)


###############################################################################
# Color palettes

core_palette = [
  '#034752', # 0
  '#02767C', # 1
  '#00A5A6', # 2
  '#11DF8C', # 3
  '#93E75E', # 4
  '#C5F1AB', # 5
  '#DCE67A', # 6
  '#FFDE5C', # 7
]

extended_palette = [
  '#47BED6', # 0
  '#1F95BD', # 1
  '#086F91', # 2
  '#034752', # 3
  '#02767C', # 4
  '#00A5A6', # 5
  '#00AF77', # 6
  '#11DF8C', # 7
  '#93E75E', # 8
  '#B9EE98', # 9
  '#FFDE5C', # 10
  '#FFB45C', # 11
  '#FF975C', # 12
]

chroma_palette = [
  '#47BED6', # 0
  '#1F95BD', # 1
  '#086F91', # 2
  '#02767C', # 3
  '#00A5A6', # 4
  '#34D2B9', # 5
  '#00AF77', # 6
  '#04D98C', # 7
  '#C5F1AB', # 8
  '#FFDE5C', # 9
  '#FFB45C', # 10
  '#FF835C', # 11
  '#FE6969', # 12
  '#FF5CAA', # 13
  '#D55CFF', # 14
  '#9A5CFF', # 15
  '#5C63FF', # 16
  '#373CC1', # 17
  '#2F1F96', # 18
  '#25084B', # 19
]

discrete_sequence_palette = [
  #'#47BED6', # 0
  '#1F95BD', # 1
  #'#086F91', # 2
  '#02767C', # 3
  #'#00A5A6', # 4
  '#34D2B9', # 5
  #'#00AF77', # 6
  #'#04D98C', # 7
  '#C5F1AB', # 8
  '#FFDE5C', # 9
  '#FFB45C', # 10
  #'#FF835C', # 11
  '#FE6969', # 12
  #'#FF5CAA', # 13
  '#D55CFF', # 14
  #'#9A5CFF', # 15
  '#5C63FF', # 16
  #'#373CC1', # 17
  #'#2F1F96', # 18
  '#25084B', # 19
]


###############################################################################
# Plotly template

pretty_template = go.layout.Template()
pretty_template.data.bar = [go.Bar(marker_color='#034752')]
pretty_template.data.scatter = [go.Scatter(marker_color='#034752')]
pretty_template.layout.bargap = 0.3

axis_color = '#5C737B'
axis_tick_label_color = '#435359'

axis_layout = dict(
    linewidth = 1,
    tickcolor = axis_color,
    linecolor = axis_color,
    tickfont = dict(
        color = axis_tick_label_color,
        size = 10,
    ),
    title_font = dict(
        size = 12,
    ),
)

pretty_template.layout.xaxis = axis_layout
pretty_template.layout.yaxis = axis_layout

pretty_template.layout.title = dict(
    font_color = '#09323A',
    font_size = 16,
    font_family = 'Messina Serif',
    x = 0,
)

pretty_template.layout.font = dict(
    family = 'Messina Sans',
)

pio.templates["pretty"] = pretty_template


###############################################################################
# Tweaks. Call them after creating the Plotly chart.

def prettify_figure(fig, highlight_ticks=False, y_label_position_y=1.15):
    fig.update_layout(template="plotly_white+pretty")

    # If the Y axis has a label, remove it and add it again as an annotation above the axis
    if fig.layout.yaxis.title.text:
        yaxis_title = fig.layout.yaxis.title.text

        fig.update_layout(yaxis_title='', yaxis_title_standoff=0)

        fig.add_annotation(
            xref="x domain",
            yref="y domain",
            x=-0.06,
            y=y_label_position_y,
            text=yaxis_title,
            showarrow=False,
            font_size=12,
        )

        fig.update_layout(margin={'t': fig.layout.margin.t + 20})

    if highlight_ticks:
        tick_color = '#CCD8D9'
        fig.update_xaxes(ticks="outside", tickcolor=tick_color)
        fig.update_yaxes(ticks="outside", tickcolor=tick_color)

    return fig

def prettify_bar_chart(fig, rotate_x_labels=True, **chart_args):
    fig = prettify_figure(fig, **chart_args)

    fig.update_xaxes(showline=True)
    fig.update_xaxes(ticks="outside")
    maxticklen = 6
    ticklen = maxticklen
    if fig.layout.height:
        ticklen = min(maxticklen, 0.02 * fig.layout.height)
    fig.update_xaxes(ticklen=ticklen)

    if rotate_x_labels:
        fig.update_xaxes(tickangle=70)

    return fig
