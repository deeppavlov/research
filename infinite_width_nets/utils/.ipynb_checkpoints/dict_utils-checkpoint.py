from collections import defaultdict


def defaultdict_to_dict(def_a: defaultdict):
    a = dict()
    for k, v in def_a.items():
        if isinstance(v, defaultdict):
            v = defaultdict_to_dict(v)
        a[k] = v
    return a


def dict_to_defaultdict(a: dict, def_a: defaultdict):
    for k, v in a.items():
        if isinstance(def_a[k], defaultdict):
            def_a[k] = dict_to_defaultdict(v, def_a[k])
        else:
            def_a[k] = v
    return def_a
