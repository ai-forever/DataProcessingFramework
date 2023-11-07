def rename_dict_keys(d: dict, keys_mapping: dict) -> dict:
    for k, v in keys_mapping.items():
        d[v] = d.pop(k)
    return d