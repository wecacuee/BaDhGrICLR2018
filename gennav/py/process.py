import os
import os.path as op
import math
import re
import operator
import glob
import numpy as np
import json
import tempfile
import subprocess
import datetime
import csv

from cStringIO import StringIO
from collections import namedtuple
from functools import partial, wraps
from itertools import product
from argparse import Namespace

conf = Namespace()
conf.full_keys = u"""apple_characters apple_prob blind chosen_map cols
        prob_shortest_per_episode prob_shortest_per_episode_std
        dist_ratio_per_episode dist_ratio_per_episode_std
        double_action_space end_time episode_length_seconds gen_stats
        goal_after_found goal_after_found_std goal_characters goal_first_found
        goal_first_found_std hostname job_id job_name jsonfile label learning
        learning_rate load_model loaded_model make_dataset make_plots maze
        method mode num_ep num_goal num_goals_std num_maps
        reward reward_std rows sourcedir spawn_characters
        start_time status test time_taken train train_test_same vars videos
        withvariations """.split()
conf.keys = u"""chosen_map reward reward_std num_goal
            prob_shortest_per_episode prob_shortest_per_episode_std
            dist_ratio_per_episode dist_ratio_per_episode_std
            num_goals_std goal_first_found goal_after_found num_maps
            goal_first_found_std goal_after_found_std sourcedir
            loaded_model vars apple_prob train_test_same label train jsonfile
            """.split()

_d_neg_one = lambda r : -1
conf.default_key_values = [
    (u"loaded_model", lambda r: r[u"chosen_map"])
    , (u"dist_ratio_per_episode", _d_neg_one)
    , (u"dist_ratio_per_episode_std", _d_neg_one)
    , (u"prob_shortest_per_episode", _d_neg_one)
    , (u"prob_shortest_per_episode_std", _d_neg_one)
    # Assume that the keys defaults are addressed in order
    , (u"train_test_same", lambda r: r[u"chosen_map"] == r["loaded_model"])]

def merge_dicts(*args):
    dicts = args
    d1 = dicts[0]
    for d2 in dicts[1:]:
        d1.update(d2)
    return d1

def str2bool(s):
    if s in "True False".split():
        return True if s == "True" else False if s == "False" else None
    else:
        raise ValueError("Bad bool {}".format(s))

def try_types(str_, types=[str2bool, int, float]):
    for t in types:
        try:
            return t(str_)
        except ValueError:
            continue
    return str_.encode('utf-8')

def filename_to_model_details(genstatjson
                              , translations = {"apples":"apple_prob"}):
    """
    >>> filename_to_model_details(
    ... "gen_stats_latest_training-1000_vars-True_apples-25.json")
    {'apple_prob': 25, 'loaded_model': 'training-1000', 'vars': True}
    """
    if genstatjson == "gen_stats.json":
        return {}

    stem = op.splitext(genstatjson)[0]
    ignorelen = len("gen_stats_latest_")
    remaining = stem[ignorelen:]
    loaded_model = remaining.split("_")[0]
    key_values = remaining.split("_")[1:]
    dict_ = {translations.get(k, k) : try_types(v)
            for k, v in [s.split("-") for s in key_values]}
    dict_.update( { "loaded_model" : loaded_model })
    return dict_

def filename_to_model_details_v2(
        genstatjson
        , patt=r"gen_stats_latest_loaded-from-(?P<loaded_model>[^_]+)_acting-on-(?P<chosen_map>[^_]+)_vars-(?P<vars>False|True)_apples-(?P<apple_prob>\d+).json"
):
    """
    >>> filename_to_model_details_v2(
    ...         "gen_stats_latest_loaded-from-training-09x09-0127_acting-on-planning-09x09-0010_vars-False_apples-25.json")
    {'apple_prob': 25, 'loaded_model': 'training-09x09-0127', 'vars': False, 'chosen_map': 'planning-09x09-0010'}
    """
    match = re.match(patt, genstatjson)
    if match:
        m_details = {k : try_types(v) for k, v in match.groupdict().items()}
    else:
        m_details = None
    return m_details

def ensure_keys_with_defaults(key_defaults, func):
    @wraps(func)
    def wrapped(sourcedir):
        for row_d in func(sourcedir):
            for k, default_from_row in key_defaults:
                row_d[k] = (row_d[k] if k in row_d else default_from_row(row_d))
            yield row_d
    return wrapped

@partial(ensure_keys_with_defaults, conf.default_key_values)
def loadmodels(sourcedir):
    return loadmodels_from_filelist(
        glob.glob(op.join(sourcedir, '*/gen_stats*.json')))

def loadmodels_from_filelist(filelist):
    """
    Loads all the json files from sourcedir
    """
    return (
        merge_dicts(
            {"sourcedir" : op.basename(op.dirname(genstat))
             , "jsonfile" : op.basename(genstat)
             , "vars" : True }
            , json.load(
                open(op.join(op.dirname(genstat), 'model_details.json' )))
            , filename_to_model_details_v2(op.basename(genstat))
            or filename_to_model_details(op.basename(genstat))
            , json.load(open(genstat)).values()[0])
        for genstat in filelist)

def select_keys(keys, from_):
    return ({k : transform(row[k]) for k, transform in keys}
            for row in from_)

def select_where(from_, where):
    return (row for row in from_
            if where(row))

def where_reduce(func, args):
    return lambda row : func(a(row) for a in args)

def where_all(*args):
    return where_reduce(all, args)

def where_any(*args):
    return where_reduce(any, args)

def where_op(opfunc, key_val):
    return lambda row: all(opfunc(row[k], v) for k, v in key_val.items())

def where_str_contains(**key_val):
    return where_op(operator.contains, key_val)

def where_equals(**equality_kw):
    return where_op(operator.eq, equality_kw)

def where_not_nan(keys):
    return lambda row: all( not math.isnan(row[k]) for k in keys )

def default_keys_transform(keys, transform=lambda v: v):
    return [((k , transform) if isinstance(k, (str, unicode)) else k)
            for k in keys]

def multigetitem(d, keys):
    return [d[k] for k in keys]

def append_to_dict_val(acc, row, keys):
    acc_keys = tuple(multigetitem(row, keys))
    acc[acc_keys] = acc.setdefault(acc_keys, []) + [row]
    return acc

def dict_diff(dicts):
    common_dict = dicts[0]
    for d in dicts[1:]:
        common_dict = {k : v for k, v in common_dict.items()
                       if k in d and d[k] == v}
        yield { k : v for k, v in d.items()
                if k not in common_dict or common_dict[k] != v}

        
def iso_strptime(datetimestr):
    return datetime.datetime.strptime(datetimestr, '%Y-%m-%dT%H:%M:%S.%f')


def latest_row(dicts):
    return max(dicts, key=lambda e: iso_strptime(e["end_time"]))

def group_by(from_, keys):
    return reduce(lambda acc, row: append_to_dict_val(acc, row, keys)
                  , from_ , dict())

def select(keys=None, from_=None, where=None):
    """
    SELECT v1 AS k1, 2*v2 AS k2 FROM table WHERE v1 = a AND v2 >= b OR v3 = c

    translates to 

    select(dict(k1=lambda r: r[v1], k2=lambda r: 2*r[v2])
        , from_=table
        , where= lambda r : r[v1] = a and r[v2] >= b or r[v3] = c)
    """
    assert from_ is not None
    idfunc = lambda k, t : t
    select_k = idfunc if keys is None  else select_keys
    if isinstance(keys, list):
        keys = default_keys_transform(keys, lambda v : v)
    idfunc = lambda t, w : t
    select_w = idfunc if where is None else select_where
    return select_k(keys, select_w(from_, where))

HData = namedtuple('HData', ['header', 'data'])
def hdata_from_dicts(dicts, header=None):
    row_d = next(dicts)
    header = header or sorted(row_d.keys())
    row_to_values = lambda r : [r[h] for h in header]
    arr_list = [row_to_values(row_d)]
    arr_list += [row_to_values(r) for r in dicts]
    return HData(header, np.array(arr_list))

def hdata_from_csv(csv_lines):
    header = next(csv_lines)
    datalines = [[row_d[h] for h in header ]
                 for row_d in csv_lines]
    return HData(header, np.asarray(datalines))

def hdata_save(csvfname, hdata):
    np.savetxt(csvfname, hdata.data, fmt='%.02f', delimiter=",",
               header=",".join(hdata.header))

def format_csv_row(row, sep, linesep):
    dummyfile = StringIO()
    csv.writer(dummyfile, delimiter=sep, lineterminator=linesep).writerow(row)
    return dummyfile.getvalue()

def format_csv_writer(dicts, sep=",", linesep="\n", header=None):
    try:
        first = next(dicts)
    except StopIteration:
        raise ValueError("Empty dicts")
    header = header or sorted(map(str, first.keys()))
    yield format_csv_row(header, sep, linesep)
    row2str = lambda r : format_csv_row(map(str, (r[h] for h in header))
                                        , sep, linesep)
    yield row2str(first)
    for row in dicts:
        yield row2str(row)

def sourcedir():
    return op.dirname(__file__) or '.'


def process(source="../exp-results/"
            , outfile="../exp-results/{}.csv"
            , labels = [
                'Static_Goal_Static_Spawn_Static_Maze'
                , 'Random_Goal_Static_Spawn_Static_Maze'
                , 'Static_Goal_Random_Spawn_Static_Maze'
                , 'Random_Goal_Random_Spawn_Static_Maze'
                , 'Random_Goal_Random_Spawn_Random_Maze']
            , keys = conf.keys
            , condition = dict(vars=True, apple_prob=0)):
    source = op.join(sourcedir(), source)
    outfile = op.join(sourcedir(), outfile)
    header = keys
    keys = default_keys_transform(
        keys,
        lambda v : "{:.02f}".format(v) if isinstance(v, float) else v)
    keys[0] = ("chosen_map", lambda cm: cm.split("-")[-1])

    # Pickup the latest results when the results are grouped by keys
    results_latest = [latest_row(values)
                        for _, values in group_by(
                                loadmodels(source),
                                "chosen_map loaded_model vars apple_prob label".split())
                      .items()
    ]
    for label in labels:
        loaded_model_part = ("training-09x09-"
                             if "Static_Maze" in label
                             else "training-1000")
        condition_label = condition.copy()
        if label == "Random_Goal_Random_Spawn_Random_Maze":
            condition_label["label"] = "Random_Goal_Random_Spawn_Static_Maze"
        else:
            condition_label["label"] = label

                                                    
        dicts = select(keys
                       , iter(results_latest)
                       , where_all(
                           where_str_contains(
                               chosen_map="training-09x09-",
                               loaded_model=loaded_model_part),
                           where_equals(**condition_label)))

        dicts = iter(sorted(dicts, key=lambda d: int(d["chosen_map"])))
        try:
            lines = list(format_csv_writer(dicts, header=header))
        except ValueError:
            print("[WARNING]: {}".format(len(list(dicts))))
            print("[WARNING]: No experiments match criteria {}"
                  .format(merge_dicts({ "label" : label}, condition)))
            continue

        with open(outfile.format(label), "w") as csvf:
            for line in lines: csvf.write(line)

def ntrain_data(source="../exp-results/"
                , outfile="../exp-results/ntrained.csv"
                , ntrain = [1, 10, 100, 500, 1000]
                , keys = conf.keys
                , not_nan_keys = []#"dist_ratio_per_episode dist_ratio_per_episode_std".split()
                , condition = {
                    "label" : "Random_Goal_Random_Spawn_Random_Maze"
                    , "chosen_map" : "testing-100" }):
    """
    variables :
    ntrain \in [1, 1000], vars \in {0, 1}, apple_prob \in {0, 25}
    """
    results_latest = (latest_row(values)
                        for _, values in group_by(
                                select(from_=loadmodels(source)
                                       , where=where_not_nan(not_nan_keys)),
                                "chosen_map loaded_model vars apple_prob label".split())
                      .items())
    dicts = select(keys
                , results_latest
                , where_all(
                    where_str_contains(loaded_model="training-")
                    , where_equals(**condition)))

    dicts = iter(sorted(
        dicts , key=lambda d: int(d["loaded_model"].split("-")[-1])))
    try:
        lines = list(format_csv_writer(dicts, header=keys))
    except ValueError:
        print("[WARNING]: No experiments match criteria {}"
            .format(condition))
        return

    with open(outfile, "w") as csvf:
        for line in lines:
            csvf.write(line)

def planning_maps_data(source="../exp-results/"
                       , outfile="../exp-results/planning_maps.csv"
                       , planning_maps = range(1, 11)
                       , keys = conf.keys
                       , constraints = {
                           "loaded_model" : "training-1000"
                           , "vars" : True
                           , "apple_prob" : 0
                           }):
    results_latest = (latest_row(values)
                        for _, values in group_by(
                                loadmodels(source),
                                "chosen_map loaded_model vars apple_prob label".split())
                      .items())
    dicts = select(keys
                , results_latest
                , where_all(
                    where_str_contains(chosen_map="planning-09x09-")
                    , where_equals(**constraints)))

    dicts = iter(sorted(
        dicts , key=lambda d: int(d["chosen_map"].split("-")[-1])))
    try:
        lines = list(format_csv_writer(dicts, header=keys))
    except ValueError:
        print("[WARNING]: No experiments match criteria {}"
            .format(constraints))
        return

    with open(outfile, "w") as csvf:
        for line in lines:
            csvf.write(line)
    


def get_img_from_video(videofile, timestamp):
    import cv2
    with tempfile.NamedTemporaryFile(suffix=".png") as tf:
        subprocess.call("ffmpeg -y -i {} -ss {} -vframes 1 {}"
                        .format(videofile, timestamp, tf.name).split())
        return cv2.imread(tf.name)


def video_to_snapshot(source
                      , outfile
                      , timestamps = [
                          "00:00:00.03"
                          , "00:00:00.57"
                          , "00:00:01.60"
                          , "00:00:06.30"
                          , "00:00:09.40"
                          , "00:00:11.40"
                          , "00:00:15.60"
                          , "00:00:19.20"
                          , "00:00:20.12"
                          , "00:00:20.70"
                      ]
                      , text_header_height = 40
                      , blue = [255, 0, 0]):
    import cv2
    total_img = []
    for ts in timestamps:
        img = get_img_from_video(source, ts)
        width = img.shape[1]
        # override first 40 rows with blue color
        img[:text_header_height, :width/3] = blue
        total_img.append(
            np.vstack((img[:, :width/3],
                       img[:, width/3:2*width/3], img[:, 2*width/3:])))
    cv2.imwrite(outfile, np.hstack(total_img))

def square_map_video_to_snapshot(source = "../exp-results/planning-09x09-0002/videos/gen_stats_latest_loaded-from-training-1000_acting-on-planning-09x09-0002_vars-True_apples-0.mp4"
                                 , outfile = None
                                 , timestamps = [
                                     "00:00:04.81"
                                     , "00:00:08.37"
                                     , "00:00:11.10"
                                     , "00:00:16.50"
                                     , "00:00:19.50"
                                     , "00:00:24.90"
                                     , "00:00:27.90"
                                     , "00:00:30.92"
                                     , "00:00:34.30"
                                     , "00:00:39.10"
                                 ]):
    video_to_snapshot(source, outfile, timestamps)


def wrench_map_video_to_snapshot(source = "../exp-results/planning-09x09-0004/videos/gen_stats_latest_loaded-from-training-1000_acting-on-planning-09x09-0004_vars-True_apples-0.mp4"
                                 , outfile = None
                                 , timestamps = [
                                     "00:00:03.21"
                                     , "00:00:06.77"
                                     , "00:00:11.70"
                                     , "00:00:15.70"
                                     , "00:00:18.90"
                                     , "00:00:22.50"
                                     , "00:00:25.99"
                                     , "00:00:29.52"
                                     , "00:00:34.70"
                                 ]):

    video_to_snapshot(source, outfile, timestamps)

def goal_map_video_to_snapshot(source = "../exp-results/planning-09x09-0006/videos/gen_stats_latest_loaded-from-training-1000_acting-on-planning-09x09-0006_vars-True_apples-0.mp4"
                                 , outfile = None
                                 , timestamps = [
                                     "00:00:06.99"
                                     , "00:00:14.27"
                                     , "00:00:16.50"
                                     , "00:00:22.30"
                                     , "00:00:23.70"
                                     , "00:00:25.70"
                                     , "00:00:27.00"
                                     , "00:00:32.30"
                                     , "00:00:39.50"
                                 ]):

    video_to_snapshot(source, outfile, timestamps)

def collect_jsons_to_csv(
        source="../exp-results/*/gen_stats_latest_*.json"
        , outfile="/tmp/one_big.csv"
        , keys = conf.full_keys
        , default_key_values = conf.default_key_values):
    dicts = partial(ensure_keys_with_defaults
                    , default_key_values)(
                        loadmodels_from_filelist)(
                            glob.glob(source))
    lines = format_csv_writer(dicts, header=keys)
    print("Writing data to big csv file: {}".format(outfile))
    with open(outfile, "w") as csvf:
        for line in lines:
            csvf.write(line)


if __name__ == '__main__':
    import sys
    func = sys.argv[1] if len(sys.argv) > 1 else "process"
    kwargs = dict()
    if len(sys.argv) > 2 and sys.argv[2]: kwargs["source"] = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]: kwargs["outfile"] = sys.argv[3]
    if len(sys.argv) > 4: raise ValueError("Unused argument {}".format(sys.argv[4]))
    globals()[func](**kwargs)
