from jua._utils.remove_none_from_dict import remove_none_from_dict


def test_remove_none_from_dict():
    assert remove_none_from_dict({"a": 1, "b": None}) == {"a": 1}

    # With list
    list_data = [{"a": 1, "b": None}, {"a": 2, "b": None}]
    full_dict = {"a": 1, "b": list_data}
    result = remove_none_from_dict(full_dict)
    assert result == {"a": 1, "b": [{"a": 1}, {"a": 2}]}

    # With nested dicts
    nested_data = {"a": 1, "b": {"c": 2, "d": None}}
    assert remove_none_from_dict(nested_data) == {"a": 1, "b": {"c": 2}}
