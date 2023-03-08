import numpy as np
import jax.numpy as jnp
from expert_mpc.expert_batch import batch_opt_nonhol
import time 

class ExpertPolicy(batch_opt_nonhol):
    """
    Expert deterministic policy running the MPC
    """

    def __init__(self):
        super().__init__()
        num = 100
        v_max = 30.0
        a_max = 8.0
        num_batch = 1000
        self.lamda_x = jnp.zeros((num_batch,  self.nvar))
        self.lamda_y = jnp.zeros((num_batch,  self.nvar))
        self.d_a = a_max * jnp.ones((num_batch, num))
        self.alpha_a = jnp.zeros((num_batch, num))
        self.alpha_v = jnp.zeros((num_batch, num))
        self.d_v = v_max * jnp.ones((num_batch, num))
        self.s_lane = jnp.zeros((num_batch, 2*num))
   
    def predict(self, obs, ax, ay, v_des):
        x = 0
        y = 0
        ub = obs[0]
        lb = obs[1]
        vx = obs[2]
        vy = obs[3]

        initial_state = jnp.hstack((x, y, vx, vy, ax, ay))
        x_obs_temp = obs[5::5]
        y_obs_temp = obs[6::5]
        vx_obs = obs[7::5]
        vy_obs = obs[8::5]
      
        x_obs, y_obs = self.compute_obs_trajectories(x_obs_temp, y_obs_temp, vx_obs, vy_obs, x, y)
        x_guess, y_guess, c_x_bar, c_y_bar, c_mean, c_cov = self.compute_traj_guess(x, y, x_obs, y_obs, v_des, lb, ub)
        start = time.time()
        c_x_best, c_y_best  = self.compute_cem(initial_state, self.lamda_x, self.lamda_y, x_obs, y_obs, x_guess, y_guess, c_x_bar, c_y_bar, lb, ub,  self.alpha_a, self.d_a, self.alpha_v, self.d_v, v_des, c_mean, c_cov, self.s_lane)
        print(f"Time Taken: {time.time() - start}")
        return np.hstack((np.array(c_x_best), np.array(c_y_best)))