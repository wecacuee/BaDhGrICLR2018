import os
import os.path as op
import itertools

import numpy as np
import matplotlib as mplab
import matplotlib.figure as mpfig
#import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf
from matplotlib.backends.backend_pgf import FigureCanvasPgf
#mplab.backend_bases.register_backend('pdf', FigureCanvasPgf)

########## Customizations

COLWIDTH = 3.3249 # Size of letter paper divided by 2
GOLDENR = 1.618
COLHEIGHT = COLWIDTH / GOLDENR

TEXT=dict(usetex=True)
FONT=dict(size=8)
SMALLERFONTSIZE=6
LINESRC=dict(markersize=2, linewidth=1)

# Symbols ahead in the list are used first
MARKERS = "+xov^s128pP*hHXdD|_<>34"
LINES = "- -- -. :".split()
COLORS = "kbmcrgy"

############ 
thisdir = op.abspath(op.dirname(__file__) or '.')
texdir = op.join(thisdir, '..')
os.environ["TEXINPUTS"] = ".:{}:{}".format(texdir, os.environ.get("TEXINPUTS", ""))

mplab.rc('text', **TEXT)
mplab.rc('font', **FONT)
mplab.rc('lines', **LINESRC)
mplab.rc('pgf', preamble=[
    r"\usepackage{times}"])

INCH2PT = lambda inch: 72*inch
PT2INCH = lambda pt : pt / 72.0
FONTSIZEIN = PT2INCH(FONT['size'])

# Generate plot styles by combining line and markers
# We should not depend on colors because they might be
# indistinguishable in print format
LINE_STYLES = [m + l for m, l in itertools.product(LINES, MARKERS)]
LINE_COLOR_PAIRS = [c + l for l, c in zip(LINE_STYLES[:len(COLORS)], COLORS)]

def subplot_box(parent_box = (0, 0, 1, 1)
                , left_margin = FONTSIZEIN*4 / COLWIDTH 
                , bottom_margin = FONTSIZEIN*4 / COLHEIGHT
                , right_margin = 0
                , top_margin = 0):
    left, bottom, width, height = parent_box
    return (left + left_margin, bottom + bottom_margin
            , width - left_margin - right_margin
            , height - bottom_margin - top_margin)

def fig_add_subplot(fig, parent_box=(0, 0, 1, 1), **kwargs):
    return ax_settings(
        fig.add_axes(subplot_box(parent_box, **kwargs)))

def ax_settings(ax):
    ax.tick_params(labelsize=SMALLERFONTSIZE)
    return ax

def reward(npzfile, outfile):
    fig = figure()
    #fig.add_subplot(111)
    # add_subplot does not do a good job
    ax = fig_add_subplot(fig, top_margin=2*FONTSIZEIN/COLHEIGHT)

    ax.set_xlabel("Training (\%)")
    ax.set_ylabel("Reward")
    ax.set_title("Mean reward per episode", **FONT)

    legends = []
    for i, n in enumerate([1, 10, 100, 1000]):
        x, y, yerr = get_reward_data(
            loaddata(
                op.join(op.dirname(npzfile), '3D-%d.npz' % n)))
        ax = plot_reward(ax, x, y, yerr, LINE_COLOR_PAIRS[i])
        legends.append('%d maps'% n)
    ax.legend(legends, ncol=2, loc='lower center', framealpha=0.2
              , fontsize=SMALLERFONTSIZE)
    fig.set_canvas(FigureCanvasPgf(fig))
    fig.savefig(outfile)

def probability(npzfile, outfile):
    fig = figure()
    # fig.add_subplot(121)
    # add_subplot does a bad job
    plot_box = subplot_box()
    l, b, w, h = plot_box
    ax = fig_add_subplot(fig, parent_box=(l, b, w*0.5, h)
                         , left_margin=0, bottom_margin=0
                         , right_margin=0, top_margin=2*FONTSIZEIN / COLHEIGHT)
    ax.set_xlabel("Training (\%)")
    ax.set_ylabel("Probability ")
    ax.set_title("P(Goals $>= 1$)", **FONT)
    ax.set_ylim((0,1))

    ax2 = fig_add_subplot(fig, parent_box=(l+w*0.5, b, w*0.5, h)
                         , left_margin=FONTSIZEIN / COLWIDTH
                          , bottom_margin=0 , right_margin=0
                          , top_margin= 2*FONTSIZEIN / COLHEIGHT)
    # Disable yticks because they are same as ax
    ax2.set_yticks([])
    ax2.set_ylim((0,1))
    ax2.set_xlabel("Training (\%)")
    ax2.set_title("P(Goals $>= 2$)", **FONT)
    legends = []
    prob_ge_one = []
    prob_ge_two = []
    for i, n in enumerate([1, 10, 100, 1000]):
        x, y1, y2 = get_probability_data(
            loaddata(
                op.join(op.dirname(npzfile), '3D-%d.npz' % n)))

        ax.plot(x, y1, LINE_COLOR_PAIRS[i])
        ax2.plot(x, y2, LINE_COLOR_PAIRS[i])
        legends.append('Maps: %d' % n)

    ax.legend(legends, loc='lower right', framealpha=0.2,
              fontsize=SMALLERFONTSIZE)
    fig.set_canvas(FigureCanvasPgf(fig))
    fig.savefig(outfile)

def get_probability_data(D):
    goal_latencies = D['goal_latencies']
    y1 = [d['p_one_goal'] for d in goal_latencies]
    y2 = [d['p_more_than_one_goal'] for d in goal_latencies]
    return D['steps'], y1, y2

def latency(npzfile, outfile):
    fig = figure()
    ax = fig.gca()
    ax.xlabel('Tranning %')
    ax.ylabel('latency 1: >1')
    legends = []
    for n in [1, 10, 100, 1000]:
        steps, latency_1, latency_gt_1 = get_latency_data(
            loaddata(op.join(op.dirname(npzfile) , '3D-%d.npz'%n)))
        ax.plot(steps, np.array(latency_1) / np.array(latency_gt_1))
        legends.append('Maps: %d' % n)
    ax.lengend()

def get_latency_data(D):
    goal_latency_data_training_steps = [
        d['more_than_one_goal_latencies'] for d in D['goal_latencies']]
    first_goal_latency = [np.mean([d[0] for d in latencies])
                           for latencies in goal_latency_data]
    second_and_more_goal_latency = [np.mean([np.mean(d[1:]) for d in latencies])
                                    for latencies in goal_latency_data]

    return D['steps'], first_goal_latency, second_and_more_goal_latency
        
    

def plot_reward(ax, x, y, yerr, style):
    ax.errorbar(x, y, yerr, fmt=style, capsize=1.5, errorevery=3)
    return ax

def loaddata(f):
    return np.load(f)
        
def get_reward_data(D):
     mean_reward = [d['mean'] for d in D['rewards']]
     std_reward = [d['std'] for d in D['rewards']]
     steps = D['steps']
     assert len(steps) == len(mean_reward)
     return steps, mean_reward, std_reward


def figure(figsize=(COLWIDTH, COLWIDTH/GOLDENR)):
    return mplab.figure.Figure(figsize=figsize)

if __name__ == '__main__':
    import sys
    func = sys.argv[1]
    source = sys.argv[2]
    outfile = sys.argv[3]
    print("TEXINPUTS {}".format(os.environ["TEXINPUTS"]))
    globals()[func](source, outfile)
    
    
