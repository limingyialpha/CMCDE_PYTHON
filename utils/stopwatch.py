from time import process_time_ns


def start_ms():
    return process_time_ns() / (10 ** 6)


def stop_ms():
    return process_time_ns() / (10 ** 6)
