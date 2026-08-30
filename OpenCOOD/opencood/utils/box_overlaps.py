# --------------------------------------------------------
# Pure-NumPy drop-in replacement for box_overlaps.pyx.
# Used on platforms without a configured C/C++ compiler (e.g. Windows
# without MSVC Build Tools) where the Cython extension cannot be built.
# Numerically equivalent to the original Cython implementation.
# --------------------------------------------------------

import numpy as np

DTYPE = np.float32


def bbox_overlaps(boxes, query_boxes):
    """
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    boxes = np.asarray(boxes, dtype=DTYPE)
    query_boxes = np.asarray(query_boxes, dtype=DTYPE)

    box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
               (query_boxes[:, 3] - query_boxes[:, 1] + 1)  # (K,)
    boxes_area = (boxes[:, 2] - boxes[:, 0] + 1) * \
                 (boxes[:, 3] - boxes[:, 1] + 1)  # (N,)

    iw = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - \
         np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1  # (N,K)
    ih = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - \
         np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1  # (N,K)

    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)

    inter = iw * ih
    ua = boxes_area[:, None] + box_area[None, :] - inter
    overlaps = np.where((iw > 0) & (ih > 0), inter / ua, 0).astype(DTYPE)

    return overlaps


def bbox_intersections(boxes, query_boxes):
    """
    For each query box compute the intersection ratio covered by boxes
    ----------
    Parameters
    ----------
    boxes: (N, 4) ndarray of float
    query_boxes: (K, 4) ndarray of float
    Returns
    -------
    overlaps: (N, K) ndarray of intersec between boxes and query_boxes
    """
    boxes = np.asarray(boxes, dtype=DTYPE)
    query_boxes = np.asarray(query_boxes, dtype=DTYPE)

    box_area = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * \
               (query_boxes[:, 3] - query_boxes[:, 1] + 1)  # (K,)

    iw = np.minimum(boxes[:, None, 2], query_boxes[None, :, 2]) - \
         np.maximum(boxes[:, None, 0], query_boxes[None, :, 0]) + 1  # (N,K)
    ih = np.minimum(boxes[:, None, 3], query_boxes[None, :, 3]) - \
         np.maximum(boxes[:, None, 1], query_boxes[None, :, 1]) + 1  # (N,K)

    iw = np.clip(iw, 0, None)
    ih = np.clip(ih, 0, None)

    intersec = np.where((iw > 0) & (ih > 0), (iw * ih) / box_area[None, :], 0).astype(DTYPE)

    return intersec


def box_vote(dets_NMS, dets_all):
    dets_NMS = np.asarray(dets_NMS, dtype=np.float32)
    dets_all = np.asarray(dets_all, dtype=np.float32)

    N = dets_NMS.shape[0]
    dets_voted = np.zeros((N, dets_NMS.shape[1]), dtype=np.float32)

    for i in range(N):
        det = dets_NMS[i, :]

        bi0 = np.maximum(det[0], dets_all[:, 0])
        bi1 = np.maximum(det[1], dets_all[:, 1])
        bi2 = np.minimum(det[2], dets_all[:, 2])
        bi3 = np.minimum(det[3], dets_all[:, 3])

        iw = bi2 - bi0 + 1
        ih = bi3 - bi1 + 1
        valid = (iw > 0) & (ih > 0)

        ua = (det[2] - det[0] + 1) * (det[3] - det[1] + 1) + \
             (dets_all[:, 2] - dets_all[:, 0] + 1) * (dets_all[:, 3] - dets_all[:, 1] + 1) - \
             iw * ih
        ov = np.where(valid, iw * ih / ua, 0)

        keep = valid & (ov >= 0.5)
        acc_box = np.sum(dets_all[keep, 4:5] * dets_all[keep, 0:4], axis=0)
        acc_score = np.sum(dets_all[keep, 4])

        dets_voted[i, 0:4] = acc_box / acc_score
        dets_voted[i, 4] = det[4]

    return dets_voted
