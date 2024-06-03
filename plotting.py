import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio


def human_format(num):
    magnitude = 0
    while abs(num) >= 1000:
        magnitude += 1
        num /= 1000.0
    # Format number to one decimal place and add a suffix
    return f'{int(num)}{["", "K", "M", "B", "T"][magnitude]}'


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


def get_cost_plot_title(estimation_method, compute_threshold_method, compute_threshold):
    if compute_threshold_method == 'window_percentile':
        title_suffix = f' for models with more compute than {compute_threshold}% of models the year before and after'
    elif compute_threshold_method == 'backward_window_percentile':
        title_suffix = f' for models with more compute than {compute_threshold}% of models the year before'
    elif compute_threshold_method == 'top_n':
        title_suffix = f' for the top-{compute_threshold} most compute-intensive ML models over time'
    elif compute_threshold_method == 'residual_from_trend':
        title_suffix = f' for the top {100 - compute_threshold}% of models farthest above compute trend'

    plot_title_lookup = {
        'cloud': 'Cloud compute cost of final training run<br>' + title_suffix,
        'hardware-acquisition': 'Acquisition cost of hardware' + title_suffix,
        'hardware-capex-energy': 'Amortized hardware CapEx + energy cost of final training run<br>' + title_suffix,
    }

    return plot_title_lookup[estimation_method]
