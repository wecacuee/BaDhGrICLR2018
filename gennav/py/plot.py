import os
import os.path as op
import itertools
import types

import numpy as np
import matplotlib as mplab
import matplotlib.figure as mpfig
#import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import FigureCanvasPdf
#from matplotlib.backends.backend_pgf import FigureCanvasPgf
#mplab.backend_bases.register_backend('pdf', FigureCanvasPdf)

from process import (process, try_types, select, where_equals, hdata_from_dicts,
                     hdata_from_csv, HData, default_keys_transform)

########## Customizations

LINEWIDTH = 6.5
COLWIDTH = 6.5 / 2 # Size of letter paper divided by 2
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

def figure(figsize=(COLWIDTH, COLWIDTH/GOLDENR)):
    fig = mplab.figure.Figure(figsize=figsize)
    fig.set_canvas(FigureCanvasPdf(fig))
    return fig

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


def csv_read(fname, sep, return_header=True):
    with open(fname) as f:
        header = f.readline().strip().split(sep)
        if return_header:
            yield header
        line = f.readline()
        while line:
            yield dict(zip(header, map(try_types, line.strip().split(sep))))
            line = f.readline()


def get_random_static_maze_spawn_goal_data(fname, sep=","):
    return csv_read(fname, sep)


def dict_get(dict_, keys):
    return [dict_[k] for k in keys]


def hdata_select_keys(hdata, keys):
    if not isinstance(hdata, HData) and isinstance(hdata, (types.GeneratorType, list)):
        hdata = hdata_from_csv(hdata)

    bool_header = [h in keys for h in hdata.header]
    return HData(keys, hdata.data[:, bool_header])


def hdata_select_rows(hdata, where):
    return HData(hdata.header, hdata.data[where(hdata), :])


def get_summary_bar_plot_data(sourcedir, labels,
                              columns = "chosen_map reward reward_std".split()):

    return [hdata_from_dicts(
        select(columns
               , csv_read(op.join(sourcedir, label) + ".csv"
                          , sep=",", return_header=False))
        , header=columns)
        for label in labels ]


def summary_bar_plot(source, outfile
                     , labels=[
                       'Static_Goal_Static_Spawn_Static_Maze'
                       , 'Random_Goal_Static_Spawn_Static_Maze'
                       , 'Static_Goal_Random_Spawn_Static_Maze'
                       , 'Random_Goal_Random_Spawn_Static_Maze'
                       , 'Random_Goal_Random_Spawn_Random_Maze'
                     ]
                     , short_labels = [
                         'Stat. Goal, Stat. Spawn, Stat. Maze'
                         , 'Rnd. Goal, Stat. Spawn, Stat. Maze'
                         , 'Stat. Goal, Rnd. Spawn, Stat. Maze'
                         , 'Rnd. Goal, Rnd. Spawn, Stat. Maze'
                         , 'Rnd. Goal, Rnd. Spawn, Rnd. Maze'
                     ]
                     , data_source = get_summary_bar_plot_data
                     , xlabel="Reward"
                     , ylabel="Map ID"
                     , xlim = [0, 240]
                     , ylim = lambda nmaps : [-0.5, nmaps -0.5]
                     , barwidth = lambda h, nmaps : h/(nmaps * 1.2)
                     , ax = fig_add_subplot(
                         figure(), top_margin=FONTSIZEIN/10.0/COLHEIGHT
                         , right_margin=FONTSIZEIN/10.0/COLWIDTH)):
    """ plot reward summary """
    if ylabel:
        ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel(xlabel)
    reward_per_exp = data_source(source, labels)
    width = barwidth(COLHEIGHT, reward_per_exp[0].data.shape[0])
    mapids = reward_per_exp[0].data[:, 0]
    if ylabel:
        ax.set_yticks(range(len(mapids)), minor=False)
        ax.set_yticklabels(map(str, map(int, list(mapids))))
        for ti in ax.yaxis.get_major_ticks():
            ti.tick1On = False
    else:
        ax.set_yticks([])

    if xlim:
        ax.set_xlim(xlim)
    ax.set_ylim([-0.5, reward_per_exp[0].data.shape[0]-0.5])
        
    legends = []
    for i, (label, reward) in enumerate(zip(short_labels, reward_per_exp)):
        reward = np.abs(reward.data)
        barc = ax.barh(np.arange(reward.shape[0])
                      + i*width - (len(reward_per_exp)-1) * width/2.0
                      , reward[:, 1], width
                      , xerr=reward[:, 2]
                       , error_kw = dict(elinewidth=0.2))
        legends.append(label)

    if outfile:
        print("saving to {}".format(outfile))
        ax.figure.savefig(outfile)
    return legends


def num_goals_summary(source, outfile, **kwargs):
    return summary_bar_plot(
        source, outfile,
        data_source = lambda src, labels: get_summary_bar_plot_data(
            src, labels
            , columns = "chosen_map num_goal num_goals_std".split())
        , xlabel="Average goal hits"
        , ylabel=None
        , xlim = [0, 24]
        , **kwargs)


def latency_from_goal_data(hdata):
    time_to_goal_first_found = hdata.data[:, 1:2]
    time_to_goal_thereafter_found = hdata.data[:, 2:3]
    time_to_goal_first_found_std = hdata.data[:, 3:4]
    time_to_goal_thereafter_found_std = hdata.data[:, 4:5]
    min_num_val = np.maximum(
        time_to_goal_first_found - time_to_goal_first_found_std, 0)
    max_num_val = time_to_goal_first_found + time_to_goal_first_found_std
    min_den_val = np.maximum(
        time_to_goal_thereafter_found - time_to_goal_thereafter_found_std, 1)
    max_den_val = time_to_goal_thereafter_found + time_to_goal_thereafter_found_std
    latency = time_to_goal_first_found / time_to_goal_thereafter_found
    min_latency = min_num_val / max_den_val
    max_latency = max_num_val / min_den_val
    latency_std = np.maximum((latency - min_latency)
                             , (max_latency - min_latency))
    return np.hstack((hdata.data[:, :1] , latency , latency_std))


def get_latency_summary_data(sourcedir, labels):
    hdata_per_exp = get_summary_bar_plot_data(
            sourcedir, labels
            , columns = "chosen_map goal_first_found goal_after_found goal_first_found_std goal_after_found_std".split())
    new_header = "chosen_map latency latency_std".split()
    return [HData(new_header, latency_from_goal_data(hdata))
            for hdata in hdata_per_exp]
    

reward_summary = summary_bar_plot


def summary_bar_plots(source, outfile):
    fig = figure(figsize=(LINEWIDTH, 0.85 * LINEWIDTH / GOLDENR))
    width, height = fig.bbox_inches.width, fig.bbox_inches.height
    plot_box = subplot_box(left_margin=4*FONTSIZEIN / width
                           , bottom_margin=3*FONTSIZEIN / height
                           , top_margin=3*FONTSIZEIN / height)
    l, b, w, h = plot_box
    ax1 = fig_add_subplot(fig, parent_box=(l, b, w*0.33, h)
                         , left_margin=0, bottom_margin=0
                         , right_margin=0, top_margin=0)
    legends = reward_summary(source, None, ax=ax1
                     , barwidth = lambda h, nmaps : h/(nmaps * 1.5))
    ax2 = fig_add_subplot(fig, parent_box=(l + w*0.33, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)
    legends = num_goals_summary(source, None, ax=ax2
                     , barwidth = lambda h, nmaps : h/(nmaps * 1.5))
    ax2.legend(legends, loc='upper right', framealpha=0.2
               , bbox_to_anchor=(1.8, 1.12)
               , fontsize=SMALLERFONTSIZE, ncol=3)

    ax3 = fig_add_subplot(fig, parent_box=(l + w*0.66, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)
    latency_summary(source, None, ax=ax3
                     , barwidth = lambda h, nmaps : h/(nmaps * 1.5))
    print("Saving to {}".format(outfile))
    fig.savefig(outfile)


def latency_summary(source, outfile, **kwargs):
    return summary_bar_plot(
        source, outfile
        , data_source = get_latency_summary_data
        , xlabel="Latency $1:>1$"
        , ylabel=None
        , xlim = [0, 2.8]
        , **kwargs)


def get_ntrain_summary_data(columns):
    return lambda source, labels : [
        hdata_from_dicts(
                select(columns
                       , csv_read(source, sep=",", return_header=False)
                       , where_equals(vars=vars_, apple_prob=apple_prob))
            , header=columns)
        for vars_, apple_prob in labels ]


def get_ntrain_latency_summary(source, labels):
    return [
        HData(
            "num_maps latency latency_std".split()
            , latency_from_goal_data(
                hdata_from_dicts(
                    select("""num_maps goal_first_found goal_after_found
                    goal_first_found_std goal_after_found_std""".split()
                           , csv_read(source, ",", return_header=False)
                           , where_equals(vars=vars_, apple_prob=apple_prob) ))))
            for vars_, apple_prob in labels
    ]
    

def ntrain_summary(source="../exp-results/ntrained.csv"
                   , outfile=None
                   , labels = [ (True, 0)
                                , (True, 25)
                                , (False, 0)
                                , (False, 25)
                   ]
                   , short_labels = ["Rnd Texture, No apples"
                                     , "Rnd Texture, With apples"
                                     , "St Texture, No apples"
                                     , "St Texture, With apples"
                   ]
                   , ):
    fig = figure(figsize=(LINEWIDTH, COLWIDTH/GOLDENR))
    width, height = fig.bbox_inches.width, fig.bbox_inches.height
    plot_box = subplot_box(left_margin=4*FONTSIZEIN/width,
                           top_margin=2*FONTSIZEIN/height,
                           bottom_margin=3*FONTSIZEIN/height)
    l, b, w, h = plot_box
    ax1 = fig_add_subplot(fig, parent_box=(l, b, w*0.33, h)
                         , left_margin=0, bottom_margin=0
                         , right_margin=0, top_margin=0)
    legends = summary_bar_plot(
        source, None
        , labels = labels 
        , short_labels = short_labels
        , data_source = get_ntrain_summary_data(
            "num_maps reward reward_std".split())
        , ylabel = "Num training maps"
        , xlim = [0, 100]
        , barwidth = lambda h, nmaps: h / (nmaps * 4)
        , ax=ax1)
    ax2 = fig_add_subplot(fig, parent_box=(l + w*0.33, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)

    legends = summary_bar_plot(
        source, None
        , labels = labels 
        , short_labels = short_labels
        , data_source = get_ntrain_summary_data(
            "num_maps num_goal num_goals_std".split())
        , ylabel = None
        , xlabel = "Average goal hits"
        , barwidth = lambda h, nmaps: h / (nmaps * 4)
        , xlim = [0, 10]
        , ax=ax2)
    ax2.legend(legends, loc='upper right', framealpha=0.2
               , bbox_to_anchor=(1.8, 1.14)
               , fontsize=SMALLERFONTSIZE, ncol=4)

    ax3 = fig_add_subplot(fig, parent_box=(l + w*0.66, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)
    legends = summary_bar_plot(
        source, None
        , labels = labels 
        , short_labels = short_labels
        , data_source = get_ntrain_latency_summary
        , ylabel = None
        , xlabel = "Latency $1:>1$"
        , barwidth = lambda h, nmaps: h / (nmaps * 4)
        , xlim=[0, 1.25]
        , ax=ax3)

    print("Saving to {}".format(outfile))
    fig.savefig(outfile)

def get_planning_maps_summary_data(columns):
    keys = default_keys_transform(columns)
    keys[0] = ("chosen_map", lambda cm : int(cm[-4:]))
    return lambda source, labels:[
        hdata_from_dicts(
                select(keys
                       , csv_read(source, sep=",", return_header=False))
            , header=columns)
        for label in labels ]

def get_planning_maps_latency_data(source, labels):
    keys = default_keys_transform("""chosen_map goal_first_found goal_after_found
                goal_first_found_std goal_after_found_std""".split())
    keys[0] = ("chosen_map", lambda cm: int(cm[-4:]))
    return [HData(
        "num_maps latency latency_std".split()
        , latency_from_goal_data(
            hdata_from_dicts(
                select(keys
                       , csv_read(source, ",", return_header=False)))))
    ]

def planning_maps_summary(source="../exp-results/planning_maps.csv"
                          , outfile=None
                          , labels = ["Random_Goal_Random_Spawn_Random_Maze"]
                          , short_labels = ["xyz"]
                          , ylabel = "Planning map ID"):
    fig = figure(figsize=(LINEWIDTH, COLWIDTH/GOLDENR))
    width, height = fig.bbox_inches.width, fig.bbox_inches.height
    plot_box = subplot_box(left_margin=4*FONTSIZEIN/width,
                           top_margin=2*FONTSIZEIN/height,
                           bottom_margin=3*FONTSIZEIN/height)
    l, b, w, h = plot_box
    ax1 = fig_add_subplot(fig, parent_box=(l, b, w*0.33, h)
                         , left_margin=0, bottom_margin=0
                         , right_margin=0, top_margin=0)
    summary_bar_plot(source, outfile
                     , labels= labels
                     , short_labels = short_labels
                     , ylabel = ylabel
                     , xlabel = "Reward"
                     , barwidth = lambda h, nmaps: h / (nmaps*0.5)
                     , xlim=[0, 110]
                     , data_source = get_planning_maps_summary_data(
                         "chosen_map reward reward_std".split())
                     , ax=ax1)
    ax2 = fig_add_subplot(fig, parent_box=(l+w*0.33, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)
    summary_bar_plot(source, outfile
                     , labels= labels
                     , short_labels = short_labels
                     , ylabel = None
                     , xlabel = "Average goal hits"
                     , barwidth = lambda h, nmaps: h / (nmaps*0.5)
                     , xlim=[0, 11]
                     , data_source = get_planning_maps_summary_data(
                         "chosen_map num_goal num_goals_std".split())
                     , ax=ax2)
    ax3 = fig_add_subplot(fig, parent_box=(l+w*0.66, b, w*0.33, h)
                         , left_margin=FONTSIZEIN/width, bottom_margin=0
                         , right_margin=0, top_margin=0)
    summary_bar_plot(source, outfile
                     , labels= labels
                     , short_labels = short_labels
                     , ylabel = None
                     , xlabel = "Latency $1: > 1$"
                     , barwidth = lambda h, nmaps: h / (nmaps*0.5)
                     , xlim=[0, 35]
                     , data_source = get_planning_maps_latency_data
                     , ax=ax3)
    fig.savefig(outfile)


def keeplotting(outfile):
    #summary_bar_plots("../exp-results", outfile)
    #ntrain_summary(op.join(thisdir, "../exp-results/ntrained.csv"), outfile)
    planning_maps_summary(op.join(thisdir, "../exp-results/planning_maps.csv"), outfile)

if __name__ == '__main__':
    # for relative imports
    import sys
    func = sys.argv[1]
    source = sys.argv[2]
    outfile = sys.argv[3]
    #print("TEXINPUTS {}".format(os.environ["TEXINPUTS"]))
    globals()[func](source, outfile)
    
    
