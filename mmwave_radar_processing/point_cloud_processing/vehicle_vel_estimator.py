import numpy as np

class VehicleVelEstimator:
    """Estimates an ego agent's velocity in x and y
    (Z computation omitted for now due to innacuracy)
    """
    def __init__(self,
                 points_per_fit=7,
                 max_iters=100,
                 fit_thresh=0.05,
                 num_close_pts=10,
                 static_vel_thresh=0.2):
        """_summary_

        Args:
            points_per_fit (int, optional): The number of points
                used to fit the model. Defaults to 7.
            max_iters (int, optional): The maximum number of iterations
                used to estimate the ego velocity. Defaults to 100.
            fit_thresh (float, optional): The square error loss threshold
                used to determine if a point is deemed to be well fit. 
                Defaults to 0.05.
            num_close_pts (int, optional): The number of well fit points
                required to assert that a given model fits. Defaults to 10.
            static_vel_thresh (float,optional): Optional threshold used to 
                identify static objects (i.e. filter out dynamic objects)
                when estimating the velocity of the ego vehicle
        """
        
        #storing the currently best it model
        self.best_fit = None
        self.best_error = np.inf

        #RANSAC parameters
        self.points_per_fit = points_per_fit
        self.max_iters=max_iters
        self.fit_thresh=fit_thresh
        self.num_close_pts=num_close_pts

        #static vel filtering
        self.static_vel_thresh = static_vel_thresh


    ####################################################################
    #Least squares estimation and fitting
    ####################################################################

    def lsq_fit_2D(self,
                   detections:np.ndarray,
                   only_2D:bool=True)->np.ndarray:
        """Use least squares estimation to estimate the 
        velocity of the vehicle

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            only_2D (bool): on True, compute the 2D velocity instead
                of the 3D velocity. Defaults to True

        Returns:
            np.ndarray: the [x,y] vor [x,y,z]elocity of the
                ego vehicle
        """
        
        #get matrix of target positions
        if only_2D:
            P = detections[:,0:2]
        else:
            P = detections[:,0:3]

        #get matrix of target velocities
        y = detections[:,3]

        #compute the normalized direction (r) vectors  
        # (p/||p|| for each row in P)
        H = np.divide(P, np.linalg.norm(P, axis=1).reshape(-1, 1))

        #estimate the velocity
        return np.linalg.inv(H.T @ H) @ H.T @ y
    

    def lsq_predict(self,
                    detections:np.ndarray,
                    v:np.ndarray) -> np.ndarray:
        """_summary_

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            v (np.ndarray): the [x,y,z] velocity or [x,y] velocity
                of the ego vehicle used 

        Returns:
            np.ndarray: Nx1 array of predicted velocity measured
                by the radar for each detection
        """


        # get matrix of target positions
        if v.shape[0] == 3: #3D case
            P = detections[:, 0:3]
        else: #2D case
            P = detections[:, 0:2]

        # compute the r vectors (p/||p|| for each row in P)
        H = np.divide(P, np.linalg.norm(P, axis=1).reshape(-1, 1))

        return H @ v

    ####################################################################
    #Loss functions
    ####################################################################
    def square_error_loss(self,v_true:np.ndarray, v_pred:np.ndarray)->np.ndarray:
        """Compute the square error loss between the predicted and actual velocity
        for a set of radar detections

        Args:
            v_true (np.ndarray): Nx1 array or the actual velocity measurement for
                each detection
            v_pred (np.ndarray): Nx1 array of the predicted velocity measurement
                for each detection

        Returns:
            np.ndarray: Nx1 array of the squared error loss between the preditions
                and the actual values
        """
        return (v_true - v_pred) ** 2
    
    def mean_square_error(self,v_true:np.ndarray,v_pred:np.ndarray)->float:
        """Compute the mean squared error loss between a prediciton and the 
            actual values

        Args:
            v_true (np.ndarray): Nx1 array or the actual velocity measurement for
                each detection
            v_pred (np.ndarray): Nx1 array of the predicted velocity measurement
                for each detection

        Returns:
            float: the mean squared error between the prediced and actual values
        """
        return np.sum(
            self.square_error_loss(v_true,v_pred)
        ) / v_true.shape[0]
    

    ####################################################################
    #Getting static detections
    ####################################################################

    def get_static_detections(self,detections:np.ndarray, ego_vel:np.ndarray)->np.ndarray:
        """For a set of detections and ego velocity estimate, get the detections corresponding
        to static objects in the environment (i.e.; non moving objects)

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            ego_vel (np.ndarray): Nx2 or Nx3 array corresponding to the velocity 
                of the ego vehicle

        Returns:
            _type_: Nx4 array of detections corresponding to static objects
        """
        #compute the predicted velocity for each detection
        env_vel = -1 * ego_vel

        v_pred = self.lsq_predict(detections,env_vel)

        #compute the square error loss
        errors = self.square_error_loss(detections[:,3],v_pred)

        #identify the errors that excede the threshold
        static_idxs = errors < self.static_vel_thresh

        return detections[static_idxs,:]

    ####################################################################
    #RANSAC estimation
    ####################################################################

    def estimate_ego_vel(self,
                   detections:np.ndarray,
                   initial_ego_vel_est:np.ndarray=np.empty(shape=0),
                   only_2D:bool=True)->np.ndarray:
        """use the RANSAC algorithm to estimate the velocity
            of the ego vehicle.

        Args:
            detections (np.ndarray): Nx4 array with the [x,y,z,vel] 
                coordinate for each detection
            initial_ego_vel_est (np.ndarray): [x,y,z] or [x,y] velocity estimate
                of the current ego velocity which will be used to 
                filter out static detections. If provided, 
                static detections will automatically be filtered out.
                If not provided, velocity will be estimated from 
                scratch. Defaults to np.empty(shape=0)
            only_2D (bool): on True, compute the 2D velocity instead
                of the 3D velocity. Defaults to True

        Returns:
            np.ndarray: the best [x,y] vor [x,y,z] velocity of the
                ego vehicle
        """

        if detections.shape[0] < self.num_close_pts:
            return np.empty(shape=0)
        
        #filter out static detections if given
        if initial_ego_vel_est.shape[0] > 0:
            detections = self.get_static_detections(
                detections=detections,
                ego_vel=initial_ego_vel_est
            )

            if detections.shape[0] < self.num_close_pts:
                return np.empty(shape=0)
        #reset the error estimate
        self.best_error = np.inf

        #initialize a random number generator
        rng = np.random.default_rng()

        for _ in range(self.max_iters):

            #get random set of permutations
            ids = rng.permutation(detections.shape[0])

            #compute an initial fit
            inlier_idxs = ids[:self.points_per_fit]
            selected_detections = detections[inlier_idxs,:]
            maybe_vel_model = self.lsq_fit_2D(
                detections=selected_detections,
                only_2D=only_2D)

            #determine additional inliers
            remaining_idxs = ids[self.points_per_fit:]
            selected_detections = detections[remaining_idxs,:]
            pred_vels = self.lsq_predict(
                detections=selected_detections,
                v=maybe_vel_model
            )
            thresholded = self.square_error_loss(
                v_true=selected_detections[:,3],
                v_pred=pred_vels
            ) < self.fit_thresh

            additional_inlier_idxs = remaining_idxs[
                np.flatnonzero(thresholded).flatten()
            ]

            inlier_idxs = np.hstack([
                inlier_idxs,
                additional_inlier_idxs])
            
            if inlier_idxs.shape[0] > self.num_close_pts:

                #re-fit the model with all of the inliers
                selected_detections = detections[inlier_idxs,:]
                maybe_vel_model = self.lsq_fit_2D(
                    detections=selected_detections,
                    only_2D=only_2D)

                pred_vels = self.lsq_predict(
                    detections=selected_detections,
                    v=maybe_vel_model
                )

                error = self.mean_square_error(
                    v_true=selected_detections[:,3],
                    v_pred=pred_vels
                )

                if error < self.best_error:
                    self.best_error = error
                    self.best_fit = maybe_vel_model

        if self.best_error != np.inf:
            return self.best_fit * -1 #estimate of velocity, not env
        else:
            return np.empty(shape=0)
    
    def get_vehicle_vel_est(self):

        return self.best_fit * -1 #estimate of velocity, not env
