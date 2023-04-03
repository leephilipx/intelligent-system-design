import numpy as np
import scipy.ndimage as ndi
import cv2

from skimage.util.dtype import dtype_limits
from skimage._shared.filters import gaussian
from skimage._shared.utils import _supported_float_type, check_nD
from skimage.feature._canny_cy import _nonmaximum_suppression_bilinear
from scipy.ndimage._filters import _gaussian_kernel1d


def _preprocess(image, mask, sigma, mode, cval):
    """Generate a smoothed image and an eroded mask.
    The image is smoothed using a gaussian filter ignoring masked
    pixels and the mask is eroded.
    Parameters
    ----------
    image : array
        Image to be smoothed.
    mask : array
        Mask with 1's for significant pixels, 0's for masked pixels.
    sigma : scalar or sequence of scalars
        Standard deviation for Gaussian kernel. The standard
        deviations of the Gaussian filter are given for each axis as a
        sequence, or as a single number, in which case it is equal for
        all axes.
    mode : str, {'reflect', 'constant', 'nearest', 'mirror', 'wrap'}
        The ``mode`` parameter determines how the array borders are
        handled, where ``cval`` is the value when mode is equal to
        'constant'.
    cval : float, optional
        Value to fill past edges of input if `mode` is 'constant'.
    Returns
    -------
    smoothed_image : ndarray
        The smoothed array
    eroded_mask : ndarray
        The eroded mask.
    Notes
    -----
    This function calculates the fractional contribution of masked pixels
    by applying the function to the mask (which gets you the fraction of
    the pixel data that's due to significant points). We then mask the image
    and apply the function. The resulting values will be lower by the
    bleed-over fraction, so you can recalibrate by dividing by the function
    on the mask to recover the effect of smoothing from just the significant
    pixels.
    """
    gaussian_kwargs = dict(sigma=sigma, mode=mode, cval=cval,
                           preserve_range=False)
    compute_bleedover = (mode == 'constant' or mask is not None)
    float_type = _supported_float_type(image.dtype)
    if mask is None:
        if compute_bleedover:
            mask = np.ones(image.shape, dtype=float_type)
        masked_image = image

        eroded_mask = np.ones(image.shape, dtype=bool)
        eroded_mask[:1, :] = 0
        eroded_mask[-1:, :] = 0
        eroded_mask[:, :1] = 0
        eroded_mask[:, -1:] = 0

    else:
        mask = mask.astype(bool, copy=False)
        masked_image = np.zeros_like(image)
        masked_image[mask] = image[mask]

        # Make the eroded mask. Setting the border value to zero will wipe
        # out the image edges for us.
        s = ndi.generate_binary_structure(2, 2)
        eroded_mask = ndi.binary_erosion(mask, s, border_value=0)

    if compute_bleedover:
        # Compute the fractional contribution of masked pixels by applying
        # the function to the mask (which gets you the fraction of the
        # pixel data that's due to significant points)
        bleed_over = gaussian(mask.astype(float_type, copy=False),
                              **gaussian_kwargs) + np.finfo(float_type).eps

    # Smooth the masked image
    smoothed_image = gaussian(masked_image, **gaussian_kwargs)

    # Lower the result by the bleed-over fraction, so you can
    # recalibrate by dividing by the function on the mask to recover
    # the effect of smoothing from just the significant pixels.
    if compute_bleedover:
        smoothed_image /= bleed_over

    return smoothed_image, eroded_mask


def canny(image, sigma=1., low_threshold=None, high_threshold=None,
          mask=None, use_quantiles=False, *, mode='constant', cval=0.0):
    
    # Regarding masks, any point touching a masked point will have a gradient
    # that is "infected" by the masked point, so it's enough to erode the
    # mask by one and then mask the output. We also mask out the border points
    # because who knows what lies beyond the edge of the image?

    check_nD(image, 2)
    dtype_max = dtype_limits(image, clip_negative=False)[1]

    if low_threshold is None:
        low_threshold = 0.1
    elif use_quantiles:
        if not(0.0 <= low_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        low_threshold /= dtype_max

    if high_threshold is None:
        high_threshold = 0.2
    elif use_quantiles:
        if not(0.0 <= high_threshold <= 1.0):
            raise ValueError("Quantile thresholds must be between 0 and 1.")
    else:
        high_threshold /= dtype_max

    if high_threshold < low_threshold:
        raise ValueError("low_threshold should be lower then high_threshold")

    # Image filtering
    smoothed, eroded_mask = _preprocess(image, mask, sigma, mode, cval)
    
    # Convolve with Gaussian kernel
    Gauss_kernel1D = _gaussian_kernel1d(sigma=sigma, order=0, radius=4*sigma)

    # Convolve with derivative of Gaussian kernel
    DoG_kernel1D = np.gradient(Gauss_kernel1D)
    pos_indices, neg_indices = np.where(DoG_kernel1D > 0), np.where(DoG_kernel1D < 0)
    DoG_kernel1D[pos_indices] = DoG_kernel1D[pos_indices] / np.sum(DoG_kernel1D[pos_indices])
    DoG_kernel1D[neg_indices] = DoG_kernel1D[neg_indices] / np.abs(np.sum(DoG_kernel1D[neg_indices]))

    # Gradient magnitude estimation (NEW: Convolution with DoG kernel)
    image64 = image.astype(np.float64)
    idog = ndi.convolve(image64, Gauss_kernel1D.reshape(1, -1), mode='nearest')
    idog = ndi.convolve(idog, DoG_kernel1D.reshape(-1, 1), mode='nearest')
    jdog = ndi.convolve(image64, Gauss_kernel1D.reshape(-1, 1), mode='nearest')
    jdog = ndi.convolve(jdog, DoG_kernel1D.reshape(1, -1), mode='nearest')
    magnitude = np.sqrt(idog * idog + jdog * jdog)

    # Gradient magnitude estimation (NEW: Derivative of Gaussian)
    # jdog = ndi.gaussian_filter(smoothed, sigma=sigma, order=[0, 1])
    # idog = ndi.gaussian_filter(smoothed, sigma=sigma, order=[1, 0])
    # magnitude = np.sqrt(idog * idog + jdog * jdog)

    # Gradient magnitude estimation
    # jsobel = ndi.sobel(smoothed, axis=1)
    # isobel = ndi.sobel(smoothed, axis=0)
    # magnitude = isobel * isobel
    # magnitude += jsobel * jsobel
    # np.sqrt(magnitude, out=magnitude)

    if use_quantiles:
        low_threshold, high_threshold = np.percentile(magnitude,
                                                      [100.0 * low_threshold,
                                                       100.0 * high_threshold])

    # Non-maximum suppression
    low_masked = _nonmaximum_suppression_bilinear(
        idog, jdog, magnitude, eroded_mask, low_threshold
    )

    # Double thresholding and edge tracking
    #
    # Segment the low-mask, then only keep low-segments that have
    # some high_mask component in them
    #
    low_mask = low_masked > 0
    strel = np.ones((3, 3), bool)
    labels, count = ndi.label(low_mask, strel)
    if count == 0:
        return low_mask

    high_mask = low_mask & (low_masked >= high_threshold)
    nonzero_sums = np.unique(labels[high_mask])
    good_label = np.zeros((count + 1,), bool)
    good_label[nonzero_sums] = True
    output_mask = good_label[labels]
    
    return output_mask
