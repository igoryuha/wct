import torch
import torch.nn.functional as F


def extract_image_patches_(image, kernel_size, strides):
    kh, kw = kernel_size
    sh, sw = strides
    patches = image.unfold(2, kh, sh).unfold(3, kw, sw)
    patches = patches.permute(0, 2, 3, 1, 4, 5)
    patches = patches.reshape(-1, *patches.shape[-3:]) # (patch_numbers, C, kh, kw)
    return patches


def style_swap(c_features, s_features, kernel_size, stride=1):

    s_patches = extract_image_patches_(s_features, [kernel_size, kernel_size], [stride, stride])
    s_patches_matrix = s_patches.reshape(s_patches.shape[0], -1)
    s_patch_wise_norm = torch.norm(s_patches_matrix, dim=1)
    s_patch_wise_norm = s_patch_wise_norm.reshape(-1, 1, 1, 1)
    s_patches_normalized = s_patches / (s_patch_wise_norm + 1e-8)
    # Computes the normalized cross-correlations.
    # At each spatial location, "K" is a vector of cross-correlations
    # between a content activation patch and all style activation patches.
    K = F.conv2d(c_features, s_patches_normalized, stride=stride)
    # Replace each vector "K" by a one-hot vector corresponding
    # to the best matching style activation patch.
    best_matching_idx = K.argmax(1, keepdim=True)
    one_hot = torch.zeros_like(K)
    one_hot.scatter_(1, best_matching_idx, 1)
    # At each spatial location, only the best matching style
    # activation patch is in the output, as the other patches
    # are multiplied by zero.
    F_ss = F.conv_transpose2d(one_hot, s_patches, stride=stride)
    overlap = F.conv_transpose2d(one_hot, torch.ones_like(s_patches), stride=stride)
    F_ss = F_ss / overlap
    return F_ss


def relu_x_1_transform(c, s, encoder, decoder, relu_target, alpha=1):
    c_latent = encoder(c, relu_target)
    s_latent = encoder(s, relu_target)
    t_features = wct(c_latent, s_latent, alpha)
    return decoder(t_features)


def relu_x_1_style_decorator_transform(c, s, encoder, decoder, relu_target, kernel_size, stride=1, alpha=1):
    c_latent = encoder(c, relu_target)
    s_latent = encoder(s, relu_target)
    t_features = style_decorator(c_latent, s_latent, kernel_size, stride, alpha)
    return decoder(t_features)


def style_decorator(cf, sf, kernel_size, stride=1, alpha=1):
    cf_shape = cf.shape
    sf_shape = sf.shape

    b, c, h, w = cf_shape
    cf_vectorized = cf.reshape(c, h * w)

    b, c, h, w = sf.shape
    sf_vectorized = sf.reshape(c, h * w)

    # map features to normalized domain
    cf_whiten = whitening(cf_vectorized)
    sf_whiten = whitening(sf_vectorized)

    # in this normalized domain, we want to align
    # any element in cf with the nearest element in sf
    reassembling_f = style_swap(
        cf_whiten.reshape(cf_shape),
        sf_whiten.reshape(sf_shape),
        kernel_size, stride
    )

    b, c, h, w = reassembling_f.shape
    reassembling_vectorized = reassembling_f.reshape(c, h*w)
    # reconstruct reassembling features into the
    # domain of the style features
    result = coloring(reassembling_vectorized, sf_vectorized)
    result = result.reshape(cf_shape)

    bland = alpha * result + (1 - alpha) * cf
    return bland


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
