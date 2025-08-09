"""
# Bipartite Sankey plots

Show how a collection of things
(people, dollars, or some such) transition from one set of categories
or groupings to another.

A minimal example

```python
weights = [
    [1.1, .9, .3],
    [2.3, .5, .7]]
sankey(weights)
```

This particular visualization is limited to data that can be represented
as a bipartite graph, that is, that connections are always between
a category from group A and a category from group B, but never between
categories within the group.
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# A colorblind-friendly palette taken from
# https://jfly.uni-koeln.de/color/
default_palette = [
    (0, 0, 0),
    (230 / 255, 159 / 255, 0),
    (86 / 255, 180 / 255, 233 / 255),
    (0, 158 / 255, 115 / 255),
    (240 / 255, 228 / 255, 66 / 255),
    (0, 114 / 255, 178 / 255),
    (213 / 255, 94 / 255, 0),
    (204 / 255, 121 / 255, 167 / 255),
]


def sankey(
    weights,
    in_labels=None,
    out_labels=None,
    in_title=None,
    out_title=None,
    fig_title=None,
    filename=None,
    title_fontsize=18,
    subtitle_fontsize=14,
    label_fontsize=11,
    dpi=600,
    fig_x=8.0,
    fig_y=8.0,
    in_palette=default_palette,
    out_palette=default_palette,
    alpha=.4,
    x_title=.08,
    x_in=.2,
    x_out=.8,
    x_label_offset=.03,
    x_val_width=.02,
    y_title=.92,
    y_subtitles=.84,
    y_flow_min=.03,
    y_flow_max=.8,
    y_val_gap=.02,
):
    """
weights: 2D NumPy array, or somthing that can be cast as one.
    Each row represents one of the initial categories, and
    each column represents one of the final categories.
    In the example above, there are two initial categories and
    three final categories.

    Each element represents the weight of the connection between
    the two categories. It can represent the amount/number of things
    that transitioned from that particular initial category and
    ended up in that particular final category.

    The weights array are the edge weights in a bipartite graph where
    each row represents an origination node and each column represents
    a termination node.

    This is the only required argument. All the others are optional and
    come with workable defaults.

in_labels: List of strings
    These are the names of the initial categories, in order,
    starting at row 0. They're shown on the left side of the plot
    by the category they represent.

out_labels: List of strings
    Like `in_labels`, but for the final categories.

in_title: String
    A label for the intitial categories as a group.

out_title: String
    A label for the final categories as a group.

fig_title: String
    A super title for the whole diagram.

filename: String
    The path and filename where a .png of the diagram will be saved.

title_fontsize, subtitle_fontsize, label_fontsize: int
    The font sizes in points for the figure title, in/out titles,
    and in/out labels, respectively.

dpi: int
    Dots-per-inch resolution of the .png.

fig_x, fig_y: float
    The x- and y-direction sizes of the diagram .png, in inches.

in_palette, out_palette: List of Matplotlib color specifications.
    What color to associate with each in/out category.
    Starting with row or column 0, cycle through the list of colors.
    If there are more categories than colors, start again at the
    beginning of the list.

    Here are all the fun ways to specify colors:
    https://matplotlib.org/stable/tutorials/colors/colors.html

alpha: float
    The transparency of the flow connectors between initial and final
    categories. Having them be somewhat transparent can help show what's
    going on when things get really tangled.

Layout parameters

When you want to nudge things around on the plot, here are all the knobs
to turn. They all fall between 0 and 1, and they all specify
an absolute position in the plot. In the x-direction 0 is the far left
and 1 is the far right. In the y-direction, 0 is the bottom and 1 is the top.
This diagram illustrates what each of them does:
https://github.com/brohrer/public-hosting/raw/main/sankey_layout_parameters.png

x_title: float
    The horizontal position of the leftmost edge of the title text.

x_val_width: float
    The width of the bars representing the initial and final categories.

x_in, x_out: float
    Horizontal positions for the centers of the bars representing the
    initial and final categories.

x_label_offset: float
    The horizontal distance between the centers of the category bars and
    the far edge of the category label text.

y_flow_min, y_flow_max: float
    The vertical positions of the bottom and top of the portion of
    the diagram that will depict the flows from initial to final categories.

y_val_gap: float
    The vertical gap between category bars.

y_title, y_subtitles: float
    The vertical positions of the centers of the title text and
    category group subtitles.
    """
    weights = np.array(weights)
    n_in, n_out = weights.shape

    # Provide some reasonable defaults.
    if in_labels is None:
        in_labels = [str(val) for val in np.arange(n_in)]
    if out_labels is None:
        out_labels = [str(val) for val in np.arange(n_out)]
    if in_title is None:
        in_title = "Original value"
    if out_title is None:
        out_title = "Final value"
    if fig_title is None:
        fig_title = "Transitions"
    if filename is None:
        filename = "sankey.png"

    # Get the figure set up.
    fig = plt.figure(figsize=(fig_x, fig_y))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # Add title and subtitles
    ax.text(
        x_title,
        y_title,
        fig_title,
        fontsize=title_fontsize,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        x_in,
        y_subtitles,
        in_title,
        fontsize=subtitle_fontsize,
        horizontalalignment="left",
        verticalalignment="center",
    )
    ax.text(
        x_out,
        y_subtitles,
        out_title,
        fontsize=subtitle_fontsize,
        horizontalalignment="right",
        verticalalignment="center",
    )

    # Figure out out to scale everything so that it fits nicely
    # in the available space.
    in_totals = np.sum(weights, axis=1)
    out_totals = np.sum(weights, axis=0)

    in_scale = (
        (y_flow_max - y_flow_min - (in_totals.size - 1) * y_val_gap) /
        np.sum(in_totals))
    out_scale = (
        (y_flow_max - y_flow_min - (out_totals.size - 1) * y_val_gap) /
        np.sum(out_totals))

    scale = np.minimum(in_scale, out_scale)
    y_flow_center = (y_flow_min + y_flow_max) / 2
    y_in_total = (
        scale * np.sum(in_totals) + (in_totals.size - 1) * y_val_gap)
    y_out_total = (
        scale * np.sum(out_totals) + (out_totals.size - 1) * y_val_gap)
    y_in_top = y_flow_center + y_in_total / 2
    y_out_top = y_flow_center + y_out_total / 2

    # Create the bars and labels on the left.
    y_in_bar_top = y_in_top
    y_in_bar_tops = []
    for i_in, in_label in enumerate(in_labels):
        # Cycle through the color palette if there are more
        # categories than colors.
        i_color = i_in % len(in_palette)
        color = in_palette[i_color]

        y_in_bar_tops.append(y_in_bar_top)
        y_in_bar_bottom = y_in_bar_top - in_totals[i_in] * scale
        left = x_in - x_val_width / 2
        right = x_in + x_val_width / 2
        top = y_in_bar_top
        bottom = y_in_bar_bottom
        bar_path = np.array([
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
        ])
        ax.add_patch(patches.Polygon(
            bar_path,
            facecolor=color,
            edgecolor=None,
        ))
        ax.text(
            x_in - x_label_offset,
            (top + bottom) / 2,
            in_label,
            fontsize=label_fontsize,
            horizontalalignment="right",
            verticalalignment="center",
        )

        y_in_bar_top = y_in_bar_bottom - y_val_gap

    # Create the bars and labels on the right.
    y_out_bar_top = y_out_top
    y_out_bar_tops = []
    for i_out, out_label in enumerate(out_labels):
        # Cycle through the color palette if there are more
        # categories than colors.
        i_color = i_out % len(out_palette)
        color = out_palette[i_color]

        y_out_bar_tops.append(y_out_bar_top)
        y_out_bar_bottom = y_out_bar_top - out_totals[i_out] * scale
        left = x_out - x_val_width / 2
        right = x_out + x_val_width / 2
        top = y_out_bar_top
        bottom = y_out_bar_bottom
        bar_path = np.array([
            [left, bottom],
            [left, top],
            [right, top],
            [right, bottom],
        ])
        ax.add_patch(patches.Polygon(
            bar_path,
            facecolor=color,
            edgecolor=None,
        ))
        ax.text(
            x_out + x_label_offset,
            (top + bottom) / 2,
            out_label,
            fontsize=label_fontsize,
            horizontalalignment="left",
            verticalalignment="center",
        )

        y_out_bar_top = y_out_bar_bottom - y_val_gap

    # Create the flow lines from left to right.
    left = x_in + x_val_width / 2
    right = x_out - x_val_width / 2
    y_in_flow_tops = y_in_bar_tops
    y_out_flow_tops = y_out_bar_tops
    for i_in, in_label in enumerate(in_labels):
        i_color = i_in % len(in_palette)
        in_color = np.array(in_palette[i_color])

        for i_out, out_label in enumerate(out_labels):
            i_color = i_out % len(out_palette)
            out_color = np.array(out_palette[i_color])

            # Calculate where each flow line needs to start and stop.
            y_in_flow_top = y_in_flow_tops[i_in]
            y_out_flow_top = y_out_flow_tops[i_out]
            dy_flow = weights[i_in, i_out] * scale
            y_in_flow_bottom = y_in_flow_top - dy_flow
            y_out_flow_bottom = y_out_flow_top - dy_flow

            # Calculate the shape of the curves.
            # Chop them finely.
            n_color_transitions = 200
            flow_top = curve(
                left, y_in_flow_top, right, y_out_flow_top,
                n=n_color_transitions)
            flow_bottom = curve(
                left, y_in_flow_bottom, right, y_out_flow_bottom,
                n=n_color_transitions)

            # Step along the length of the curves that outline
            # the top and bottom of the flow ribbon.
            # Create the effect of a smooth color transition.
            # by making a large number of slim polygon slices, each with a
            # slightly different color.
            for i in range(n_color_transitions - 1):
                flow_path = [
                    [flow_top[i, 0], flow_top[i, 1]],
                    [flow_top[i + 1, 0], flow_top[i + 1, 1]],
                    [flow_bottom[i + 1, 0], flow_bottom[i + 1, 1]],
                    [flow_bottom[i, 0], flow_bottom[i, 1]],
                ]
                in_color_weight = np.cos(
                    np.pi * i / (2 * (n_color_transitions - 2))) ** 2
                blend_color = (
                    in_color_weight * in_color +
                    (1 - in_color_weight) * out_color)
                ax.add_patch(patches.Polygon(
                    flow_path,
                    facecolor=blend_color,
                    edgecolor=None,
                    alpha=alpha,
                ))

            y_in_flow_tops[i_in] = y_in_flow_bottom
            y_out_flow_tops[i_out] = y_out_flow_bottom

    # Remove all the figure edge splines, axes, ticks, and labels.
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.tick_params(bottom=False, top=False, left=False, right=False)
    ax.tick_params(
        labelbottom=False, labeltop=False, labelleft=False, labelright=False)

    plt.savefig(filename, dpi=dpi)


def curve(x0, y0, x1, y1, n=100):
    """
    Generate a sinusoidal curve that smoothly transitions from
    (x0, y0) to (x1, y1) in n equal steps along the x axis.

    Returns
    2D NumPy array with n rows and 2 columns
    Column 0 contains the x values and column 1 the y values.
    """
    x = np.linspace(x0, x1, n)
    y = y0 + (y1 - y0) * np.sin(np.pi * (x - x0) / (2 * (x1 - x0))) ** 2
    return np.concatenate((x[:, np.newaxis], y[:, np.newaxis]), axis=1)


if __name__ == "__main__":
    """
    When run as a script, this module runs a little demonstration of
    a Sankey diagram.
    """
    filename = "sankey_test.png"
    scale = 10
    n_vals = 4
    weights = np.random.sample(size=(n_vals, n_vals)) * scale
    in_labels = [
        f"Starting label {str(val)}" for val in np.arange(n_vals)[::-1]]
    out_labels = [
        f"Ending label {str(val)}" for val in np.arange(n_vals)[::-1]]

    sankey(
        weights,
        in_labels=in_labels,
        out_labels=out_labels,
        in_title="How it started",
        out_title="How it's going",
        fig_title="Sankey demonstration",
        filename=filename,
    )
    print(f"There's a Sankey demo plot stored in {filename}")
