def check_time_steps(t_int, t_trim_min, t_trim_max):

    if t_trim_min % t_int != 0:
        raise ValueError(
            "The t_trim_min = {} must be a multiple of t_int = {}".format(
                t_trim_min, t_int
            )
        )
    if t_trim_max % t_int != 0:
        raise ValueError(
            "The t_trim_max = {} must be a multiple of t_int = {}".format(
                t_trim_max, t_int
            )
        )
