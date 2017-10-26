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

def where_not(where):
    return lambda row: not where(row)

def where_not_nan(keys):
    return lambda row: all( not math.isnan(row[k]) for k in keys )

def default_keys_transform(keys, transform=lambda v: v):
    return [((k , transform) if isinstance(k, (str, unicode)) else k)
            for k in keys]

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
