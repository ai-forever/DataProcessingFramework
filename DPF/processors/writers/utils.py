from typing import Dict, Any


def rename_dict_keys(d: Dict[Any, Any], keys_mapping: Dict[Any, Any]) -> Dict[Any, Any]:
    for k, v in keys_mapping.items():
        d[v] = d.pop(k)
    return d
