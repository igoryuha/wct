import torch
import numpy as np


def relu_x_1_transform(c, s, encoder, decoder, relu_target, alpha=1):
    c_latent = encoder(c, relu_target)
    s_latent = encoder(s, relu_target)
    t_features = wct(c_latent, s_latent, alpha)
    return decoder(t_features)


def wct(cf, sf, alpha=1):
    cf_shape = cf.shape

    b, c, h, w = cf_shape
    cf_vectorized = cf.reshape(c, h*w)

    b, c, h, w = sf.shape
    sf_vectorized = sf.reshape(c, h*w)

    cf_transformed = whitening(cf_vectorized)
    cf_transformed = coloring(cf_transformed, sf_vectorized)

    cf_transformed = cf_transformed.reshape(cf_shape)

    bland = alpha * cf_transformed + (1 - alpha) * cf
    return bland


def feature_decomposition(x):
    x_mean = x.mean(1, keepdims=True)
    x_center = x - x_mean
    x_cov = x_center.mm(x_center.t()) / (x_center.size(1) - 1)

    e, d, _ = torch.svd(x_cov)
    d = d[d > 0]
    e = e[:, :d.size(0)]

    return e, d, x_center, x_mean


def whitening(x):
    e, d, x_center, _ = feature_decomposition(x)

    transform_matrix = e.mm(torch.diag(d ** -0.5)).mm(e.t())
    return transform_matrix.mm(x_center)


def coloring(x, y):
    e, d, _, y_mean = feature_decomposition(y)

    transform_matrix = e.mm(torch.diag(d ** 0.5)).mm(e.t())
    return transform_matrix.mm(x) + y_mean
