import numpy as np


def normalize(mat):
    max_value = np.max(mat)
    return mat / max_value


def Md(Id, N, gz):
    r = 16  # constante
    sigma = 0.3  # constante

    Fd = Id - np.sum(np.sum(N)) / N.size

    Mfront = np.maximum(Fd, np.zeros_like(Fd))
    Mmid = np.maximum(1 - r * Fd ** 2, np.zeros_like(Fd))
    Mback = np.maximum(-Fd, np.zeros_like(Fd))

    if gz > -1 and gz < sigma:
        return Mfront
    elif gz > -sigma and gz < sigma:
        return Mmid
    elif gz > sigma and gz < 1:
        return Mback


def Mf(H, G, I, alpha=6):
    """
        H     : (hx, hy) head position
        G     : (gx, gy) gaze target direction
        I     : image
        alpha : parameter to decide the angle of view

        return : FOV attention Map
    """

    FOV = np.zeros_like(I)

    for i in range(I.shape[0]):
        for j in range(I.shape[1]):
            theta = np.arccos(((np.array([i, j]) - H) * G) / (np.linalg.norm(np.array([i, j]) - H) * np.linalg.norm(G)))
            FOV[i, j] = np.max(1 - ((alpha * theta) / np.pi), 0)

    return FOV


def DAM(depth_maps, imgs_heads, gz, H, G, I, alpha=6):
    for i in range(np.size(depth_maps)):
        heat_maps = Md(depth_maps[i], imgs_heads[i], gz[i]) * Mf(H[i], G[i], I[i],
                                                                 alpha)  # on doit multiplier par le module FOV

    return heat_maps