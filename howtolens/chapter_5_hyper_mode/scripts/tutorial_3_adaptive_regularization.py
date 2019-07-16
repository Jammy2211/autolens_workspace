# Regularization:

# The issue with regularization is more subtle. When we regularizate a source reconstruction, we compare the
# difference between reconstructed flux between every source pixel and its neighbors. The amount by which we use
# these differences to penalize our solution (e.g. reduce the likelihood and evidence) is scaled by the
# regularization coefficient. Remember, we want to choose a regularization coefficient such that our source
# reconstruction fits the data well, but don't over-fit noise in the image.

# The reason we fail to reconstruct the central regions of the lensed source galaxy is because it is being
# over-smooothed, that is, we are using too high of a regularization coefficient. If we were to reduce the
# regularization coefficient
