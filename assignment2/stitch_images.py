import numpy as np

def compute_affine_transform(src_pts, dst_pts):
    src_pts_h = np.hstack((src_pts, np.ones((len(src_pts), 1))))
    dst_pts_h = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
    aff_matrix, _, _, _ = np.linalg.lstsq(src_pts_h, dst_pts_h, rcond=None)
    aff_matrix = np.reshape(aff_matrix, (3, 3))
    return aff_matrix

def compute_projective_transform(src_pts, dst_pts):
    src_pts_h = np.hstack((src_pts, np.ones((len(src_pts), 1))))
    dst_pts_h = np.hstack((dst_pts, np.ones((len(dst_pts), 1))))
    proj_matrix, _, _, _ = np.linalg.lstsq(src_pts_h, dst_pts_h, rcond=None)
    proj_matrix = np.reshape(proj_matrix, (3, 3))
    return proj_matrix

def ransac(src_kp, dst_kp, iterations, min_samples, threshold):
    best_model = None
    best_inliers = []

    for _ in range(iterations):
        sample_indices = np.random.choice(
            len(src_kp), min_samples, replace=False)
        sample_src = src_kp[sample_indices]
        sample_dst = dst_kp[sample_indices]

        model = compute_affine_transform(sample_src, sample_dst)
        distances = np.linalg.norm(dst_kp - np.dot(src_kp, model.T), axis=1)
        inliers = np.where(distances < threshold)[0]

        if len(inliers) > len(best_inliers):
            best_model = model
            best_inliers = inliers

    return best_model, best_inliers
