import numpy as np

def get_rot_matrix(rot_angle_rad):
    """Get a rotation matrix for a given rotation angle

    Args:
        rot_angle_rad (double): the desired rotation angle in radians

    Returns:
        np.ndarray: a 2x2 rotation matrix for the desired rotation
    """
    return np.array([
        [np.cos(rot_angle_rad), -1.0 * np.sin(rot_angle_rad)],
        [np.sin(rot_angle_rad), np.cos(rot_angle_rad)]
    ])

def get_angle_from_rot_matrix(rot_matrix:np.ndarray):
    """Returns the rotation angle from a given rotation matrix

    Args:
        rot_matrix (np.ndarray): the rotation matrix

    Returns:
        np.float64: the rotation angle in radians
    """
    return np.arctan2(rot_matrix[1,0],rot_matrix[0,0])

def apply_rot_trans(points:np.ndarray,rot_angle_rad,trans):
        """Apply a rotation and translation to a set of points using
        the formulat out = (points * R.T) + trans

        Args:
            points (np.ndarray): Nx2 array of points to be transformed
            rot_angle_rad (_type_): the rotation angle in radians to rotate
                the points by
            trans (_type_): the [x,y] translation vector to apply

        Returns:
            np.ndarray: An Nx2 array of the transformed points
        """

        # transform 'points' (using the calculated rotation and translation)
        R = get_rot_matrix(rot_angle_rad)
        
        #apply the rotation matrix and translation
        return (points @ R.T) + trans    

def apply_multiple_rot_trans(
          points:np.ndarray,
          rot_angles_rad,
          translations):
        """Generate a set of N new point clouds using N different
        rotations and translations with the formula
        out = (points * R.T) + trans

        Args:
            points (np.ndarray): Mx2 array of M points to be transformed
            rot_angle_rad (np.ndarray): Nx1 array of rotations (radians)
            translations (np.ndarray): the Nx2 array of translation vectors
              to apply

        Returns:
            np.ndarray: An Nx2 array of the transformed points
        """

        #compute the rotation matrix for each rotation
        R = np.array([get_rot_matrix(rot_angle_rad) for \
        rot_angle_rad in rot_angles_rad])

        #extend the points array to be 1xMx2
        points_extended = points[np.newaxis,:,:]

        #transpose each of the rotation matricies (output is Nx2x2)
        R_transposed = R.transpose((0,2,1))

        #extend the translation array to be Nx1x2
        translations_extended = translations[:,np.newaxis,:]

        #apply the rotation matrix and translation
        return np.matmul(points_extended,R_transposed) + translations_extended

def apply_unique_rot_trans_to_multiple_points(
          points:np.ndarray,
          rot_angles_rad,
          translations):
        """For a set of N unique rotations and translations
        and N points. Performs the nth rotation and translation
        on the nth point  with the formula
        out = (point * R.T) + trans

        Args:
            points (np.ndarray): Mx2 array of N points to be transformed
            rot_angle_rad (np.ndarray): Nx1 array of rotations (radians)
            translations (np.ndarray): the Nx2 array of translation vectors
              to apply

        Returns:
            np.ndarray: An Nx2 array of the transformed points
        """

        assert points.shape[0] == \
            rot_angles_rad.shape[0] == \
                translations.shape[0], \
                "all inputs must have N rows"
        
        #compute the rotation matrix for each rotation
        R = np.array([get_rot_matrix(rot_angle_rad) for \
        rot_angle_rad in rot_angles_rad])

        #extend the points array to be 1xMx2
        points_extended = points[:,np.newaxis,:]

        #transpose each of the rotation matricies (output is Nx2x2)
        R_transposed = R.transpose((0,2,1))

        #extend the translation array to be Nx1x2
        translations_extended = translations[:,np.newaxis,:]

        #apply the rotation matrix and translation
        return np.matmul(points_extended,R_transposed) + translations_extended

def wrap_heading(heading_rad):
    """wraps the heading to be between [-pi,pi]

    Args:
        heading_rad (_type_): the heading in radians

    Returns:
        _type_: the wrapped heading in radians
    """
    #implement wrapping around when abs(heading) > pi
    if np.abs(heading_rad) > np.pi:
        return (-1 * np.sign(heading_rad)
            * (2 * np.pi - np.abs(heading_rad))
        ) 

    else:
        return heading_rad