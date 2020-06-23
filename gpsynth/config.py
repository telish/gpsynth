# Use regression to ensure good continuation. This makes the first and last
# sample of a wavetable to be close. Regression is not used when using periodic
# kernels or when doing waveshaping (regardless of the configuration).
good_continuation_regression = True
