import os


def is_main_process():
    try:
        return int(os.environ["LOCAL_RANK"]) == 0
    except:
        return True


def main_print(*args, **kwargs):
    if is_main_process():
        print(*args, **kwargs)