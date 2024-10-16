import scipy
import paddle
import math
import crepe


###############################################################################
# Pitch unit conversions
###############################################################################


def bins_to_cents(bins):
    """Converts pitch bins to cents"""
    cents = crepe.CENTS_PER_BIN * bins + 1997.3794084376191

    # Trade quantization error for noise
    return dither(cents)


def bins_to_frequency(bins):
    """Converts pitch bins to frequency in Hz"""
    return cents_to_frequency(bins_to_cents(bins))


def cents_to_bins(cents, quantize_fn=math.floor):
    """Converts cents to pitch bins"""
    bins = (cents - 1997.3794084376191) / crepe.CENTS_PER_BIN
    return quantize_fn(bins)


def cents_to_frequency(cents):
    """Converts cents to frequency in Hz"""
    return 10 * 2 ** (cents / 1200)


def frequency_to_bins(frequency, quantize_fn=math.floor):
    """Convert frequency in Hz to pitch bins"""
    return cents_to_bins(frequency_to_cents(frequency), quantize_fn)


def frequency_to_cents(frequency):
    """Convert frequency in Hz to cents"""
    return 1200 * math.log2(frequency / 10.)


# ###############################################################################
# # Utilities
# ###############################################################################


def dither(cents):
    """Dither the predicted pitch in cents to remove quantization error"""
    noise = scipy.stats.triang.rvs(c=0.5,
                                   loc=-crepe.CENTS_PER_BIN,
                                   scale=2 * crepe.CENTS_PER_BIN,
                                   size=cents.shape)
    # return cents + cents.new_tensor(noise)
    return cents + paddle.to_tensor(noise, dtype=cents.dtype, stop_gradient=cents.stop_gradient)
