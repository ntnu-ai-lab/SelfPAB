import numpy as np
import pandas as pd
import scipy.signal

_RESAMPLERS = {}


def get_resampler(resampler):
    """
    Get a resampling function by name.
    """
    if callable(resampler):
        return resampler
    if not resampler in _RESAMPLERS:
        raise Exception(f'No resampler with name "{resampler}" is registered')
    return _RESAMPLERS[resampler]


def get_resampler_names():
    """
    Get all resampling function names.
    """
    return sorted(_RESAMPLERS)


def resampler(name=None):
    """
    Register a resampler, optionally under a specific name.
    """
    def decorator(f):
        _RESAMPLERS[name or f.__name__] = f
        return f

    return decorator


@resampler('fourier')
def resample_fourier(data, num_out):
    """
    A very nice, general purpose resampling algorithm
    based on taking the fourier transform of the input
    signal, cropping or padding frequencies to desired
    rate, and inversing the transform to any num_out.
    """
    return scipy.signal.resample(data, num_out)


@resampler('decimate')
def resample_decimate(data, num_out):
    """
    Downsamples using decimate. Require an integer downsample factor,
    so it is particularly unsuited when resampling iterators of varying
    size or when using padding.
    """
    downsample_factor = len(data) / num_out
    if not downsample_factor == int(downsample_factor):
        raise Exception(f'Resample factor must be an integer for decimate resampling. '
                        f'Received len(data)={len(data)}, num_out={num_out} factor={downsample_factor}')
    res = scipy.signal.decimate(data, int(downsample_factor), zero_phase=True)
    return res[:num_out]


_PADDERS = {}


def get_padder(padder):
    """
    Get a padder by name.
    """
    if callable(padder):
        return padder
    if not padder in _PADDERS:
        raise Exception(f'No padder with name "{padder}" is registered')
    return _PADDERS[padder]


def get_padder_names():
    """
    Get all padder names.
    """
    return sorted(_PADDERS)


def padder(name=None):
    """
    Decorator for registering a padder.
    """
    def decorator(f):
        _PADDERS[name or f.__name__] = f
        return f

    return decorator


@padder('reflect')
def reflect_padder(window, n):
    """
    Reflect edges.
    """
    return window[1:n + 1][::-1], \
           window[len(window) - n - 1:len(window) - 1][::-1]


@padder('inv_reflect')
def inv_reflect_padder(window, n):
    """
    Reflect edges, but also negate their sign.
    """
    return (window[n] - window[:n])[::-1], \
           (2 * window[-1] - window[len(window) - n - 1:len(window) - 1])[::-1]


@padder('wrap')
def wrap_padder(window, n):
    """
    Wrap edges like if the signal was periodic.
    """
    return window[len(window) - n:], \
           window[:n]


@padder('repeat')
def repeat_padder(window, n):
    """
    Repeat the first/last value.
    """
    return np.zeros(n) + window[0], \
           np.zeros(n) + window[-1]


@padder('zero')
def zero_padder(window, n):
    """
    Set padded values to zero.
    """
    return np.zeros(n), np.zeros(n)


def resample_real(window, num_out, resampler_f, padder_f=None, pad_size=0):
    """
    Resample real a real valued time series using the provided
    padding and resampling strategy.

    Inputs:
    - window : pd.Series[float]
      The the real-valued time series that will be resampled
    - num_out : int
      How many samples that should be in the resampled signal
    - resampler_f : function( pd.Series[float], int ) -> pd.Series[float]
      Resampling function. Should accept input signal and desired
      output size and return a resampled series
    - padder_f : function( pd.Series[float], int ) -> (pd.Series[float], pd.Series[float])
      Padding function. Should accept input signal and pad amount
      and return pre and post padding. Padding will NOT be applied
      if the argument is left as None
    - pad_size : int
      Amount of padding in input signal

    Returns:
      A resampled pd.Series[float]
    """
    window = np.asarray(window)
    num_in = len(window)
    # Pad window
    if padder_f:
        # Compute full amounts
        num_pad_in = min(num_in - 1, pad_size or 0)
        num_pad_out = round(num_pad_in * num_out / num_in)
        num_in_padded = num_pad_in + num_in + num_pad_in
        num_out_padded = num_pad_out + num_out + num_pad_out
        # Create expanded input array
        data_in = np.zeros(num_in_padded)
        # Set content
        data_in[num_pad_in:num_in_padded - num_pad_in] = window[:]
        pad_before, pad_after = padder_f(window, num_pad_in)
        data_in[0:num_pad_in] = pad_before[:]
        data_in[num_in_padded - num_pad_in:] = pad_after[:]
    else:
        num_in_padded = num_in
        num_out_padded = num_out
        data_in = window
        # Apply resampler
    data_out = resampler_f(data_in, num_out_padded)
    # Remove padding
    if padder_f:
        return data_out[num_pad_out:num_out_padded - num_pad_out]
    else:
        return data_out


def resample_timestamp(window, num_out):
    """
    Resample a timestamp column by linearly interpolating timestamps

    Inputs:
    - window : pd.Series[pd.Timestamp]
      The timestamp series that will be resampled
    - num_out : int
      Desired number of samples in output

    Returns:
      A pd.Series[pd.Timestamp] with resampled timestamps
    """
    # Make sure window is a netural pandas series with (0,n) index
    window = pd.Series(window).reset_index(drop=True)
    # Get first and last timestamp
    t0, t1 = window.iloc[0], window.iloc[-1]
    # Compute (equally spaced) timestamp deltas in input and output
    num_in = len(window)
    step_in = (t1 - t0) / (num_in - 1)
    step_out = step_in * (num_in / num_out)
    # Generate equally spaced timestamp that will start at 00
    return t0 + pd.Series(np.arange(num_out)) * step_out


def resample_discrete(window, num_out):
    """
    Resample a discrete column by taking closest value

    Inputs:
    - window : pd.Series[??]
      Series of any datatype that will be resampled
    - num_out : int
      Desired number of samples in output

    Returns:
      A pd.Series[??] with resampled objects
    """
    window = pd.Series(window).reset_index(drop=True)
    num_in = len(window)
    step_size = num_in / num_out
    indices = np.round(step_size * np.arange(num_out)).clip(0, num_in - 1)
    return window.iloc[indices.astype(int)].reset_index(drop=True)


def resample_it(df_it, source_rate, target_rate,
                resampler, padder=None, pad_size=None,
                discrete_columns=[], timestamp_columns=[]):
    """
    Resample an iterator of dataframes.
    """
    # Compute resampling factor (i.e. frequency scale: 2=resample to double frequency)
    resampling_factor = target_rate / source_rate
    # Get resampler and padder function
    resampler_f = get_resampler(resampler)
    padder_f = get_padder(padder) if padder else None
    i = 0
    for df in df_it:
        # Compute in and out sizes
        num_in = len(df)
        num_out = round(num_in * resampling_factor)
        # Break if too few datapoints
        if num_in < 2 or num_out < 2:
            try:
                # It should only be ok if it is at the end of the iterator
                next(df_it)
                raise Exception(f'Resample dimensions too small num_in={num_in}<2 or num_out={num_out}<2.')
            except StopIteration:
                break
        # Create output df
        result = pd.DataFrame(index=range(i, num_out))
        i += num_out
        # Resample each column
        # print(f'Start resampling using {resampler}')
        for column in df.columns:
            # Apply resampler based on column type
            if column in discrete_columns:
                result[column] = resample_discrete(df[column], num_out)
            elif column in timestamp_columns:
                result[column] = resample_timestamp(df[column], num_out)
            else:
                result[column] = resample_real(df[column], num_out, resampler_f, padder_f, pad_size)
        # Yield result
        yield result


def resample(df, source_rate, target_rate,
             resampler, padder=None, pad_size=None,
             discrete_columns=[], timestamp_columns=[]):
    """
    Resample a whole dataframe.
    """
    return next(resample_it(iter([df]), source_rate, target_rate,
                            resampler, padder, pad_size,
                            discrete_columns, timestamp_columns))
