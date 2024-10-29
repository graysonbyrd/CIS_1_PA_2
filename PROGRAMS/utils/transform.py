import numpy as np


class FT:
    """Class for defining Frame Transformations."""

    def __init__(self, R: np.ndarray, t: np.ndarray):
        assert R.shape == (3, 3)
        assert t.shape == (1, 3)
        self.R = R  # rotation matrix
        self.t = t  # translation vector

    def transform_pts(self, pts: np.ndarray):
        """Transforms a set of points in one coordinate frame to another.

        Params:
            pts (np.ndarray): set of points in one coordinate frame

        Returns:
            np.ndarray: Transformed points in the target coordinate frame
        """
        if pts.shape[1] != 3:
            raise ValueError(f"Points must be of shape (n, 3), not (n, {pts.shape[1]})")
        return np.dot(self.R, pts.T).T + self.t

    def inverse_transform_pts(self, pts: np.ndarray):
        """Uses the inverse of the transform to transform a set of points from
        one coordinate frame to another.

        Params:
            pts (np.ndarray): set of points in one coordinate frame

        Returns:
            np.ndarray: Transformed points in the target coordinate frame
        """
        if pts.shape[1] != 3:
            raise ValueError(f"Points must be of shape (n, 3), not (n, {pts.shape[0]})")
        return np.dot(self.R.T, pts.T).T - np.dot(self.R.T, self.t.T).T
