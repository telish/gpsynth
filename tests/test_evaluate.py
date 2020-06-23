from evaluate import speed_test, make_continuous_discontinuous
import plots


def test_speed_test():
    speed_test(n_wavetables=1)


def test_make_continuous_discontinuous(tmp_path: str):
    make_continuous_discontinuous(tmp_path)


def test_plots(tmp_path: str):
    plots.main(tmp_path)
