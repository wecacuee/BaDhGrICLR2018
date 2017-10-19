import os
import os.path as op
import operator
import glob
import numpy as np
import json
from collections import namedtuple
from functools import partial
from itertools import product
from argparse import Namespace

conf = Namespace()
conf.keys = """chosen_map reward reward_std num_goal
            num_goals_std goal_first_found goal_after_found num_maps
            goal_first_found_std goal_after_found_std sourcedir
            loaded_model vars apple_prob train_test_same label train jsonfile
            """.split()

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
    return str_

def filename_to_model_details(genstatjson
                              , translations = {"apples":"apple_prob"}):
    """
    >>> filename_to_model_details(
    ... "gen_stats_latest_training-1000_vars-True_apples-25.json")
    {'loaded_model': 'training-1000', 'apple_prob': True, 'training': True, 'vars': True, 'train_test_same' : False}
    """
    if genstatjson == "gen_stats.json":
        return {"train_test_same" : True }

    stem = op.splitext(genstatjson)[0]
    ignorelen = len("gen_stats_latest_")
    remaining = stem[ignorelen:]
    loaded_model = remaining.split("_")[0]
    key_values = remaining.split("_")[1:]
    dict_ = {translations.get(k, k) : try_types(v)
            for k, v in [s.split("-") for s in key_values]}
    dict_.update( { "loaded_model" : loaded_model, "train_test_same" : False })
    return dict_

def loadmodels(sourcedir):
    return (
        merge_dicts(
            {"sourcedir" : op.basename(op.dirname(genstat))
             , "jsonfile" : op.basename(genstat) }
            , json.load(
                open(op.join(op.dirname(genstat), 'model_details.json' )))
            , filename_to_model_details(op.basename(genstat))
            , json.load(open(genstat)).values()[0])
        for genstat in glob.glob(op.join(sourcedir, '*/gen_stats*.json')))

def select_keys(keys, from_):
    return ({k : transform(row.get(k, "")) for k, transform in keys}
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
    return lambda row: all(opfunc(row.get(k, ""), v) for k, v in key_val.items())

def where_str_contains(**key_val):
    return where_op(operator.contains, key_val)

def where_equals(**equality_kw):
    return where_op(operator.eq, equality_kw)

def default_keys_transform(keys, transform=lambda v: v):
    return [(k , transform)  for k in keys]

def multigetitem(d, keys):
    return [d[k] for k in keys]

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
    if isinstance(keys, list) and isinstance(keys[0], (str, unicode)):
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
    return HData(header, datalines)

def hdata_save(csvfname, hdata):
    np.savetxt(csvfname, hdata.data, fmt='%.02f', delimiter=",",
               header=",".join(hdata.header))

def format_csv_writer(dicts, sep=",", linesep="\n", header=None):
    try:
        first = next(dicts)
    except StopIteration:
        raise ValueError("Empty dicts")
    header = header or sorted(first.keys())
    yield sep.join(header) + linesep
    row2str = lambda r : sep.join(map(str, (r[h] for h in header))) + linesep
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
                       , loadmodels(source)
                       , where_all(
                           where_str_contains(
                               chosen_map="training-09x09-",
                               loaded_model=loaded_model_part),
                           where_equals(**condition_label)))

        dicts = iter(sorted(dicts, key=lambda d: int(d["chosen_map"])))
        try:
            lines = list(format_csv_writer(dicts, header=header))
        except ValueError:
            print("[WARNING]: No experiments match criteria {}"
                  .format(merge_dicts({ "label" : label}, condition)))
            continue

        with open(outfile.format(label), "w") as csvf:
            for line in lines: csvf.write(line)

def ntrain_data(source="../exp-results/"
                , outfile="../exp-results/ntrained.csv"
                , ntrain = [1, 10, 100, 500, 1000]
                , keys = conf.keys
                , condition = {
                    "label" : "Random_Goal_Random_Spawn_Random_Maze"}):
    """
    variables :
    ntrain \in [1, 1000], vars \in {0, 1}, apple_prob \in {0, 25}
    """
    dicts = select(keys
                , loadmodels(source)
                , where_all(
                    lambda r : r["chosen_map"] == r.get("loaded_model","")
                    , where_str_contains(loaded_model="training-")
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

if __name__ == '__main__':
    import sys
    func = sys.argv[1] if len(sys.argv) > 1 else "process"
    kwargs = dict()
    if len(sys.argv) > 2 and sys.argv[2]: kwargs["source"] = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]: kwargs["outfile"] = sys.argv[3]
    globals()[func](**kwargs)
