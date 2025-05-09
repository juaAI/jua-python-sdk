def remove_none_from_dict(d: dict, max_recursion: int = 10) -> dict:
    if max_recursion == 0:
        return d
    cleaned_dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned_dict[k] = remove_none_from_dict(v, max_recursion - 1)
        elif isinstance(v, list):
            cleaned_list = []
            for item in v:
                if isinstance(item, dict):
                    cleaned_list.append(remove_none_from_dict(item, max_recursion - 1))
                else:
                    cleaned_list.append(item)
            cleaned_dict[k] = cleaned_list
        elif v is None:
            continue
        else:
            cleaned_dict[k] = v
    return cleaned_dict
