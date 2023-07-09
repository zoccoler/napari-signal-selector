def generate_categorical_labels_colormap(categorical_labels):
    """
    Generate a colormap for the categorical (prediction) labels.
    Parameters
    ----------
    categorical_labels : np.ndarray
        categorical_labels array.
    Returns
    -------
    cmap_dict : dict
        Dictionary mapping each categorical label (predictions) to a respective color.

    """
    import numpy as np
    import colorcet as cc
    from matplotlib import colormaps

    cmap = colormaps['cet_glasbey_category10']

    # generate dictionary mapping each category to its respective color
    # list cycling with  % introduced for all labels
    cmap_dict = {
        int(category): (
            cmap(int(category) - 1)
            if category > 0
            else [0, 0, 0, 0]  # background label
        )
        for category in np.unique(categorical_labels)
    }
    # # take care of background label
    # cmap_dict[0] = [0, 0, 0, 0]
    return cmap_dict
