import os
import os.path as op
import numpy as np
import json
from collections import namedtuple
from functools import partial

def merge_dicts(d1, d2):
    d1.update(d2)
    return d1

def loadmodels(sourcedir):
    return (
        merge_dicts(
            json.load(
                open(op.join(sourcedir, d, 'model_details.json' )))
            , json.load(
                open(op.join(sourcedir, d, 'gen_stats.json'))).values()[0])
        for d in os.listdir(sourcedir)
        if op.isdir(op.join(sourcedir, d)))

def select_keys(keys, from_):
    return ({k : transform(row.get(k, "")) for k, transform in keys}
            for row in from_)

def select_where(from_, where):
    return (row for row in from_
            if where(row))

def where_from_dict(**equality_kw):
    return lambda row : all(row[k] == v for k, v in equality_kw.items())

def default_keys_transform(keys, transform=lambda v: v):
    return [(k , transform)  for k in keys]

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
    idfunc = lambda t, w : t
    select_w = idfunc if where is None else select_where
    return select_k(keys, select_w(from_, where))

HData = namedtuple('HData', ['header', 'data'])
def hdata_from_dicts(dicts):
    row_d = next(dicts)
    header = sorted(row_d.keys())
    row_to_values = lambda r : [r[h] for h in header]
    arr_list = [row_to_values(row_d)]
    arr_list += [row_to_values(r) for r in dicts]
    return HData(header, np.array(arr_list))

def hdata_save(csvfname, hdata):
    np.savetxt(csvfname, hdata.data, fmt='%.02f', delimiter=",",
               header=",".join(hdata.header))

def dicts_csv_writer(csvf, dicts, sep=",", linesep="\n", header=None):
    first = next(dicts)
    header = header or sorted(first.keys())
    csvf.write(sep.join(header) + linesep)
    row2str = lambda r : sep.join(map(str, (r[h] for h in header))) + linesep
    csvf.write(row2str(first))
    for row in dicts:
        csvf.write(row2str(row))

def sourcedir():
    return op.dirname(__file__) or '.'


def process(source="../exp-results/"
            , outfile="../exp-results/{}.csv"
            , labels = [
                'Static_Goal_Static_Spawn_Static_Maze'
                #,'Random_Goal_Static_Spawn_Static_Maze'
                ,'Static_Goal_Random_Spawn_Static_Maze'
                ,'Random_Goal_Random_Spawn_Static_Maze'
                , 'Random_Goal_Random_Spawn_Random_Maze']
            , keys = """chosen_map reward reward_std num_goal
            num_goals_std goal_first_found goal_after_found num_maps
            goal_first_found_std goal_after_found_std""".split()):
    source = op.join(sourcedir(), source)
    outfile = op.join(sourcedir(), outfile)
    header = keys
    keys = default_keys_transform(
        keys,
        lambda v : "{:.02f}".format(v) if isinstance(v, float) else v)
    keys[0] = ("chosen_map", lambda cm: cm[-4:])
    for label in labels[:-1]:
        dicts = select(keys
                       , loadmodels(source)
                       , where_from_dict(label=label))
        dicts = iter(sorted(dicts, key=lambda d: d["chosen_map"]))
        with open(outfile.format(label), "w") as csvf:
            dicts_csv_writer(csvf, dicts, header=header)
        #hdata = hdata_from_dicts(dicts)
        #average = HData(hdata.header,
        #                np.mean(hdata.data, axis=0, keepdims=True))
        #hdata_save(outfile.format(label), average)

    label = labels[-1]
    dicts = select([keys[-1]] + keys[1:-1]
                    , loadmodels(source)
                    , where_from_dict(label=label))
    dicts = iter(sorted(dicts, key=lambda d: d["num_maps"]))
    with open(outfile.format(label), "w") as csvf:
        dicts_csv_writer(csvf, dicts, header=[header[-1]] + header[1:-1])


if __name__ == '__main__':
    import sys
    func = sys.argv[1] if len(sys.argv) > 1 else "process"
    kwargs = dict()
    if len(sys.argv) > 2 and sys.argv[2]: kwargs["source"] = sys.argv[2]
    if len(sys.argv) > 3 and sys.argv[3]: kwargs["outfile"] = sys.argv[3]
    globals()[func](**kwargs)
