import numpy as np


def label_img_colors(img, colors):
    """Labels pixels according to their closest color.

    Args:
        img (np.array): input image to label.
        colors (np.array): input colors in RGB.

    Returns:
        np.array where the 3rd dimension corresponds to the number of colors.
    """
    # Find the euclidian distance between each pixel and each color
    norms = np.array([np.linalg.norm(img - color, axis=2)
                      for color in colors])
    labeled_pixels = np.argmin(norms, axis=0)   # Assigns labels to pixels
    # One-hot encode labels. Shape becomes (STATE_H, STATE_W, NUM_LABELS)
    return np.eye(len(colors))[labeled_pixels]

def label_distribution_from_block(arr, start, end):
    """Returns the distribution of labels in a block from arr.

    If start or end are floats, the distribution includes weighted elements
    on the edge of the block.

    Args:
        arr (np.ndarray): 3D array from which to obtain the block where the
            3rd dimension is a one-hot encoded vector of labels.
        start (tuple of floats): top left corner of the block.
        end (tuple of floats): bottom right corner of block.

    Returns:
        Array containing the distribution of labels.
    """
    assert len(start) == len(end) == 2
    assert len(arr.shape) == 3
    start = np.array(start)
    end = np.array(end)

    start_idx = np.ceil(start).astype(np.int)
    end_idx = np.floor(end).astype(np.int)

    counts = np.sum(arr[start_idx[0]:end_idx[0], start_idx[1]:end_idx[1]],
                    axis=(0, 1))

    NUM_DECIMALS = 7    # Prevents floating point errors
    top_weight, left_weight = np.round(-start % 1, NUM_DECIMALS)
    bottom_weight, right_weight = np.round(end, NUM_DECIMALS) % 1

    counts = counts.astype(np.float)

    # Edges not including corners
    if top_weight > 1e-4:   # Tolerate floating point errors with small numbers
        counts += top_weight * \
                np.sum(arr[start_idx[0] - 1, start_idx[1]:end_idx[1]], axis=0)
    if bottom_weight > 1e-4:
        counts += bottom_weight * \
                np.sum(arr[end_idx[0], start_idx[1]:end_idx[1]], axis=0)
    if left_weight > 1e-4:
        counts += left_weight * \
                np.sum(arr[start_idx[0]:end_idx[0], start_idx[1] - 1],
                       axis=0)
    if right_weight > 1e-4:
        counts += right_weight * \
                np.sum(arr[start_idx[0]:end_idx[0], end_idx[1]],
                       axis=0)
    # Corners
    if top_weight and left_weight:
        counts += top_weight * left_weight * arr[start_idx[0], start_idx[1]]
    if top_weight and right_weight:
        counts += top_weight * right_weight * arr[start_idx[0], end_idx[1]]
    if bottom_weight and left_weight:
        counts += bottom_weight * left_weight * arr[end_idx[0], start_idx[1]]
    if bottom_weight and right_weight:
        counts += bottom_weight * right_weight * arr[end_idx[0], end_idx[1]]

    return counts / np.sum(counts)


def to_low_res_truth(labeled_img, shape, nsamples=None):
    """Calculates the distrubution of each region.

    Uses the area of intersection between each pixel and the larger low-res
    region to weight each label in order to construct the distribution.

    Args:
        labeled_img (np.ndarray): 3D array from which to obtain the block
            where the 3rd dimension is a one-hot encoded vector of labels.
        shape (tuple of length 2): output resolution.
        nsamples: dummy argument to mantain API consistency.

    Returns:
        Low resolution array where the 3rd dimension contains the distribution
        of labels present within the high resolution elements corresponding
        to the location in the low resolution array.
    """
    assert len(labeled_img.shape) == 3
    original_shape = labeled_img.shape
    if original_shape[2] == 1:
        # Add another "negative" layer:
        labeled_img = np.concatenate([labeled_img, 1 - labeled_img], axis=2)

    row_step, col_step = np.divide(labeled_img.shape[0:2], shape)

    low_res = np.array([[label_distribution_from_block(
                            labeled_img, (i, j), (i + row_step, j + col_step))
                         for j in np.arange(0, labeled_img.shape[1], col_step)]
                        for i in np.arange(0, labeled_img.shape[0], row_step)])

    if original_shape[2] == 1:
        # Remove the "negative" layer
        low_res = np.expand_dims(low_res[:, :, 0], 2)

    return low_res


def sample_low_res_region(labeled_img, top_left, bottom_right, nsamples=100):
    """Samples from the distribution of labels in a region of arr.

    Args:
        arr (np.ndarray): 3D array from which to obtain the block where the
            3rd dimension is a one-hot encoded vector of labels.
        start (tuple of floats): top left corner of the region.
        end (tuple of floats): bottom right corner of region.
        nsamples (int): number of samples to take.

    Returns:
        Array containing the distribution of labels.
    """
    idxs = np.random.uniform(top_left, bottom_right, (nsamples, 2))
    idxs = idxs.T.astype(np.int)
    samples = labeled_img[idxs[0], idxs[1]]
    counts = np.sum(samples, axis=0)
    return counts / np.sum(counts)


def to_low_res_sample_regions(labeled_img, shape, nsamples_per_region=1000):
    """Samples from within each region to approximate a region's distribution.

    May be faster than calculating the true distribution of labels.

    Args:
        labeled_img (np.ndarray): 3D array from which to obtain the block
            where the 3rd dimension is a one-hot encoded vector of labels.
        shape (tuple of length 2): output resolution.
        nsamples_per_region (int): number of samples to take for each lower
            resolution pixel.

    Returns:
        Low resolution array where the 3rd dimension contains the distribution
        of labels present within the high resolution elements corresponding
        to the location in the low resolution array.
    """
    assert len(labeled_img.shape) == 3

    row_step, col_step = np.divide(labeled_img.shape[0:2], shape)

    low_res = np.array([[sample_low_res_region(
                            labeled_img, (i, j), (i + row_step, j + col_step),
                            nsamples_per_region)
                         for j in np.arange(0, labeled_img.shape[1], col_step)]
                        for i in np.arange(0, labeled_img.shape[0], row_step)])

    return low_res


def to_low_res_sample_frame(labeled_img, shape, nsamples=1000):
    """Samples from the entire frame to approximate each region's distribution.

    May be faster than calculating the true distribution of labels.

    Args:
        labeled_img (np.ndarray): 3D array from which to obtain the block
            where the 3rd dimension is a one-hot encoded vector of labels.
        shape (tuple of length 2): output resolution.
        nsamples_per_region (int): number of samples to take for each lower
            resolution pixel.

    Returns:
        Low resolution array where the 3rd dimension contains the distribution
        of labels present within the high resolution elements corresponding
        to the location in the low resolution array.
    """
    assert len(labeled_img.shape) == 3

    row_step, col_step = np.divide(labeled_img.shape[0:2], shape)

    counts = np.zeros((shape[0], shape[1], 1))    # Use ones to make robust?
    sums = np.zeros((shape[0], shape[1], labeled_img.shape[2]))
    idxs = np.random.uniform((0, 0), shape[0:2], (nsamples, 2)).astype(np.int)
    for idx in map(tuple, idxs):
        counts[idx] += 1
        sums[idx] += labeled_img[idx]
    return sums / counts


def to_low_res_polar(
        labeled_image, origin, angles, distances, nsamples_per_region=1000):
    """Samples label distributions for the regions bounded by angles and distances.

    Outputs a (len(distances) - 1, len(angles) - 1) array where the index of
    the closest distance bin falls in row len(distances) - 1 and the index of
    the furthest distance bin falls in row 0.

    Bins for angles correspond to boundaries between columns, including the edges.
    Negative angles are to the left of the car and positive angles are to the
    right of the car.

    Args:
        labeled_image: raw labeled image data.
        origin: location of the origin in the labeled image.
        angles: boundary angles of polar regions. Expected to be ordered
            smallest to largest. Negative angles are to the left of the car
            and positive angles are to the right of the car.
        distances: boundary distances of polar regions. Expected to be ordered
            smallest to largest.
        nsamples_per_region: number of samples taken from each region.

    Returns:
        np.ndarray containing the distribution of labels in each region.
    """
    distribution = np.zeros(
            (len(distances) - 1, len(angles) - 1, labeled_image.shape[2]))
    for i, (r_min, r_max) in enumerate(zip(distances[-2::-1], distances[:0:-1])):
        for j, (angle_min, angle_max) in enumerate(zip(angles[:-1], angles[1:])):
            # Sample polar coordinate
            rs = np.random.uniform(r_min, r_max, nsamples_per_region)
            sampled_angles = np.random.uniform(
                    angle_min, angle_max, nsamples_per_region)
            # Convert to cartesian
            sampled_rows = - rs * np.cos(sampled_angles) + origin[0]
            sampled_cols = rs * np.sin(sampled_angles) + origin[1]
            # Convert to int for indexing
            sampled_rows = np.round(sampled_rows).astype(np.int)
            sampled_cols = np.round(sampled_cols).astype(np.int)
            # Mask sampled rows/cols that are outside the image
            mask_rows = (0 < sampled_rows) & (sampled_rows < labeled_image.shape[0])
            mask_cols = (0 < sampled_cols) & (sampled_cols < labeled_image.shape[1])
            mask = mask_rows * mask_cols
            # Clip samples to within image
            sampled_rows = np.clip(sampled_rows, 0, labeled_image.shape[0] - 1)
            sampled_cols = np.clip(sampled_cols, 0, labeled_image.shape[1] - 1)
            # Create distribution over the region
            samples = labeled_image[sampled_rows, sampled_cols] * mask[:, None]
            counts = np.sum(samples, axis=0)
            distribution[i, j] = counts / nsamples_per_region

    return distribution
