from argparse import Namespace


def pretty_print_ns(ns, indent=0):
    prefix = " " * indent
    if isinstance(ns, dict):
        for k, v in ns.items():
            print(f"{prefix}{k}:")
            pretty_print_ns(v, indent + 2)
    elif hasattr(ns, "__dict__"):  # Namespace
        for k, v in vars(ns).items():
            print(f"{prefix}{k}:")
            pretty_print_ns(v, indent + 2)
    else:
        print(f"{prefix}{ns}")

def deep_merge(base: dict, override: dict) -> dict:
    """Function for deep merging two dictionaries."""
    result = base.copy()
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
        ):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def dict_to_namespace(d: dict) -> Namespace:
    ns = Namespace()
    for k, v in d.items():
        if isinstance(v, dict):
            setattr(ns, k, dict_to_namespace(v))
        else:
            setattr(ns, k, v)
    return ns