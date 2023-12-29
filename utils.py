debug = True  # Set this to False if you don't want to print debug messages


def debug_print(*args, **kwargs):
    if debug:
        print(*args, **kwargs)