def make_legend(ax, c1, c2, changefont=False):
    """
    Make a legend with background color c1, edge color c2 and optionally a user selected font size
    """
    if not changefont:
        legend = ax.legend(frameon=True)
    else:
        legend = ax.legend(frameon=True, prop={'size': changefont})

    frame = legend.get_frame()
    frame.set_facecolor(c1)
    frame.set_edgecolor(c2)