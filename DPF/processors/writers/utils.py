from typing import Any


def rename_dict_keys(d: dict[Any, Any], keys_mapping: dict[Any, Any]) -> dict[Any, Any]:
    for k, v in keys_mapping.items():
        d[v] = d.pop(k)
    return d
