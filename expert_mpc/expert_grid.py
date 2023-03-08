import jax
import numpy as np
import jax.numpy as jnp
from functools import partial
from jax import jit, random
from utils import bernstein_coeff_order10_arbitinterval

class batch_opt_nonhol():

	def __init__(self, maxiter_proj):

		self.v_max = 30.0 
		self.v_min = 0.1
		self.a_max = 8.0
		self.num_obs = 10
		self.num_batch = 1000 # 1000
		self.steer_max = 0.6
		self.kappa_max = 0.230
		self.wheel_base = 2.5
		self.a_obs = 6.0 ########### rectangle of car and ego: length = 4, width = 1.4. We over approximate a_obs and b_obs to account for orientation change of obstacle and ego.
		self.b_obs = 3.2
		
		self.t_fin = 15
		self.num = 100
		self.t = self.t_fin/self.num
		self.ellite_num =  50 # 50
		self.ellite_num_projection = 150 # 150

		self.v_pert = 16  #############3 how much the desired velocity needs to be perturbed to generate intial guess trajectory
		self.initial_up_sampling = 30

		tot_time = np.linspace(0, self.t_fin, self.num)
		self.tot_time = tot_time
		tot_time_copy = tot_time.reshape(self.num, 1)

		
		self.P, self.Pdot, self.Pddot = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, tot_time_copy[0], tot_time_copy[-1], tot_time_copy)
		self.P_jax, self.Pdot_jax, self.Pddot_jax = jnp.asarray(self.P), jnp.asarray(self.Pdot), jnp.asarray(self.Pddot)
		self.nvar = jnp.shape(self.P_jax)[1]

		
		###############################

		# self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
	
		self.A_eq_x = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0]   ))
		self.A_eq_y = jnp.vstack(( self.P_jax[0], self.Pdot_jax[0], self.Pddot_jax[0], self.Pdot_jax[-1] ))
					
		self.A_vel = self.Pdot_jax 
		self.A_acc = self.Pddot_jax
		self.A_projection = jnp.identity(self.nvar)
		
		###################################3 obstacle avoidance		
		self.A_y_centerline = self.P_jax
		self.A_obs = jnp.tile(self.P_jax, (self.num_obs, 1))
		self.A_lane = jnp.vstack(( self.P_jax, -self.P_jax    ))
		
		key = random.PRNGKey(0)

		self.key = key

		####################################
		##########################3 computing initial covariance for trajectories

		A = np.diff(np.diff(np.identity(self.num), axis = 0), axis = 0)

		temp_1 = np.zeros(self.num)
		temp_2 = np.zeros(self.num)
		temp_3 = np.zeros(self.num)
		temp_4 = np.zeros(self.num)

		temp_1[0] = 1.0
		temp_2[0] = -2
		temp_2[1] = 1
		temp_3[-1] = -2
		temp_3[-2] = 1
		temp_4[-1] = 1.0

		A_mat = -np.vstack(( temp_1, temp_2, A, temp_3, temp_4   ))
		# A_mat = A

		R = np.dot(A_mat.T, A_mat)
		mu = np.zeros(self.num)
		cov = np.linalg.pinv(R)
		################# Gaussian Trajectory Sampling
		eps_k = np.random.multivariate_normal(mu, 0.1*cov, (int(self.num_batch/1), ))
		
		self.cov = jnp.asarray(cov)
		# self.eps_k = jnp.asarray(self.eps_k)
		
		###################### parameters

		self.rho_nonhol = 1.0
		self.rho_ineq = 1
		self.rho_obs = 100
		self.rho_projection = 1.0
		self.rho_goal = 1.0
		self.rho_lane = 1.0

		#################################################
		self.maxiter = maxiter_proj # 100
		self.maxiter_cem = 1

		self.k_p = 20
		self.k_d = 2.0*jnp.sqrt(self.k_p)

		self.k_p_1 = 4
		self.k_d_1 = 2.0*jnp.sqrt(self.k_p_1)

		self.k_p_v = 20
		self.k_d_v = 2.0*jnp.sqrt(self.k_p_v)
  
		################################ up sampling for polynomials

		self.num_up = 1500
		self.t_up = self.t_fin/self.num_up

		tot_time_up = np.linspace(0, self.t_fin, self.num_up)
		self.tot_time_up = tot_time_up.reshape(self.num_up, 1)
		# tot_time_copy = tot_time.reshape(num, 1)

		self.P_up, self.Pdot_up, self.Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10, self.tot_time_up[0], self.tot_time_up[-1], self.tot_time_up)

		self.P_jax_up, self.Pdot_jax_up, self.Pddot_jax_up = jnp.asarray(self.P_up), jnp.asarray(self.Pdot_up), jnp.asarray(self.Pddot_up)

		################################################################################

		self.alpha_mean = 0.6
		self.alpha_cov = 0.6

		self.lamda = 0.9
		self.vec_product = jit(jax.vmap(self.comp_prod, 0, out_axes=(0)))

		######################################################################################## matrices for computing initial guess based on neural parameters

		self.rho_v = 1 
		self.rho_offset = 1

		self.P_jax_1 = self.P_jax[0:25, :]
		self.P_jax_2 = self.P_jax[25:50, :]
		self.P_jax_3 = self.P_jax[50:75, :]
		self.P_jax_4 = self.P_jax[75:100, :]

		self.Pdot_jax_1 = self.Pdot_jax[0:25, :]
		self.Pdot_jax_2 = self.Pdot_jax[25:50, :]
		self.Pdot_jax_3 = self.Pdot_jax[50:75, :]
		self.Pdot_jax_4 = self.Pdot_jax[75:100, :]
			
		self.Pddot_jax_1 = self.Pddot_jax[0:25, :]
		self.Pddot_jax_2 = self.Pddot_jax[25:50, :]
		self.Pddot_jax_3 = self.Pddot_jax[50:75, :]
		self.Pddot_jax_4 = self.Pddot_jax[75:100, :]

		self.num_partial = 25
		
		self.num_initial_sampling = 30*self.num_batch
  
		self.weight_smoothness = 0.1
		self.cost_smoothness = self.weight_smoothness*jnp.dot(self.Pddot_jax.T, self.Pddot_jax)
  
	@partial(jit, static_argnums=(0,))	
	def sampling_param(self, y_lane_lb, y_lane_ub, mean_param, cov_param, v_des):

		v_des_temp = jnp.linspace(v_des-self.v_pert, v_des+self.v_pert, 20) # This part is hardcoded 
		y_des_temp = jnp.linspace(y_lane_lb, y_lane_ub, 50)

		v_des_samples, y_des_samples = jnp.meshgrid(v_des_temp, y_des_temp)

		v_des_samples = v_des_samples.reshape(self.num_batch, )
		y_des_samples = y_des_samples.reshape(self.num_batch, )

		v_des_1 = v_des_samples
		v_des_2 = v_des_samples
		v_des_3 = v_des_samples
		v_des_4 = v_des_samples

		y_des_1 = y_des_samples
		y_des_2 = y_des_samples
		y_des_3 = y_des_samples
		y_des_4 = y_des_samples
		
		neural_output_batch = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4 )).T

		return neural_output_batch

	@partial(jit, static_argnums=(0,))	
	def compute_x_guess(self, initial_state, neural_output_batch):

		x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state


		x_init_vec = x_init*jnp.ones((self.num_batch, 1))
		y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

		vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
		vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

		ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
		ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

		b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
		b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_batch, 1)) ))


		v_des_1 = neural_output_batch[:, 0]
		v_des_2 = neural_output_batch[:, 1]
		v_des_3 = neural_output_batch[:, 2]
		v_des_4 = neural_output_batch[:, 3]

		y_des_1 = neural_output_batch[:, 4]
		y_des_2 = neural_output_batch[:, 5]
		y_des_3 = neural_output_batch[:, 6]
		y_des_4 = neural_output_batch[:, 7]


		##########################################################################################
		A_pd_1 = self.Pddot_jax_1-self.k_p*self.P_jax_1-self.k_d*self.Pdot_jax_1
		b_pd_1 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_1)[:, jnp.newaxis]
		
		A_pd_2 = self.Pddot_jax_2-self.k_p*self.P_jax_2-self.k_d*self.Pdot_jax_2
		b_pd_2 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_2)[:, jnp.newaxis]
			
		A_pd_3 = self.Pddot_jax_3-self.k_p*self.P_jax_3-self.k_d*self.Pdot_jax_3
		b_pd_3 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_3)[:, jnp.newaxis]
		
		A_pd_4 = self.Pddot_jax_4-self.k_p*self.P_jax_4-self.k_d*self.Pdot_jax_4
		b_pd_4 = -self.k_p*jnp.ones((self.num_batch, self.num_partial ))*(y_des_4)[:, jnp.newaxis]
		


		A_vd_1 = self.Pddot_jax_1-self.k_p_v*self.Pdot_jax_1#-self.k_d_v*self.Pddot_jax_1
		b_vd_1 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_1)[:, jnp.newaxis]


		A_vd_2 = self.Pddot_jax_2-self.k_p_v*self.Pdot_jax_2#-self.k_d_v*self.Pddot_jax_2
		b_vd_2 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_2)[:, jnp.newaxis]

		A_vd_3 = self.Pddot_jax_3-self.k_p_v*self.Pdot_jax_3#-self.k_d_v*self.Pddot_jax_3
		b_vd_3 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_3)[:, jnp.newaxis]

		A_vd_4 = self.Pddot_jax_4-self.k_p_v*self.Pdot_jax_4#-self.k_d_v*self.Pddot_jax_4
		b_vd_4 = -self.k_p_v*jnp.ones((self.num_batch, self.num_partial ))*(v_des_4)[:, jnp.newaxis]

		
		##############################################

		cost_x = self.cost_smoothness+self.rho_v*jnp.dot(A_vd_1.T, A_vd_1)+self.rho_v*jnp.dot(A_vd_2.T, A_vd_2)+self.rho_v*jnp.dot(A_vd_3.T, A_vd_3)+self.rho_v*jnp.dot(A_vd_4.T, A_vd_4)
		cost_y = self.cost_smoothness+self.rho_offset*jnp.dot(A_pd_1.T, A_pd_1)+self.rho_offset*jnp.dot(A_pd_2.T, A_pd_2)+self.rho_offset*jnp.dot(A_pd_3.T, A_pd_3)+self.rho_offset*jnp.dot(A_pd_4.T, A_pd_4)
		
		cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
		cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
		
		lincost_x = -self.rho_v*jnp.dot(A_vd_1.T, b_vd_1.T).T-self.rho_v*jnp.dot(A_vd_2.T, b_vd_2.T).T-self.rho_v*jnp.dot(A_vd_3.T, b_vd_3.T).T-self.rho_v*jnp.dot(A_vd_4.T, b_vd_4.T).T
		lincost_y = -self.rho_offset*jnp.dot(A_pd_1.T, b_pd_1.T).T-self.rho_offset*jnp.dot(A_pd_2.T, b_pd_2.T).T-self.rho_offset*jnp.dot(A_pd_3.T, b_pd_3.T).T-self.rho_offset*jnp.dot(A_pd_4.T, b_pd_4.T).T 
	
		sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
		sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T


		#######################3

		primal_sol_x = sol_x[:,0:self.nvar]
		primal_sol_y = sol_y[:,0:self.nvar]

		c_x = primal_sol_x[0]
		c_y = primal_sol_y[0]

		x = jnp.dot(self.P_jax, primal_sol_x.T).T
		xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
		xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T

		y = jnp.dot(self.P_jax, primal_sol_y.T).T
		ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
		yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T


		return primal_sol_x, primal_sol_y

	@partial(jit, static_argnums=(0,))	
	def compute_obs_trajectories(self, x_obs_temp, y_obs_temp, vx_obs_temp, vy_obs_temp, x_init, y_init):

		dist_obs = (x_init-x_obs_temp)**2+(y_init-y_obs_temp)**2

		idx_sort = jnp.argsort(dist_obs)

		x_obs_sort = x_obs_temp[idx_sort[0:self.num_obs]]
		y_obs_sort = y_obs_temp[idx_sort[0:self.num_obs]]

		vx_obs_sort  = vx_obs_temp[idx_sort[0:self.num_obs]]
		vy_obs_sort  = vy_obs_temp[idx_sort[0:self.num_obs]]
		
		x_obs = x_obs_sort+vx_obs_sort*self.tot_time[:, jnp.newaxis]
		y_obs = y_obs_sort+vy_obs_sort*self.tot_time[:, jnp.newaxis]

		x_obs = x_obs.T 
		y_obs = y_obs.T 

		return x_obs, y_obs

	@partial(jit, static_argnums=(0,))	
	def compute_boundary_vec(self, x_init, vx_init, ax_init, y_init, vy_init, ay_init):

		x_init_vec = x_init*jnp.ones((self.num_batch, 1))
		y_init_vec = y_init*jnp.ones((self.num_batch, 1)) 

		vx_init_vec = vx_init*jnp.ones((self.num_batch, 1))
		vy_init_vec = vy_init*jnp.ones((self.num_batch, 1))

		ax_init_vec = ax_init*jnp.ones((self.num_batch, 1))
		ay_init_vec = ay_init*jnp.ones((self.num_batch, 1))

		b_eq_x = jnp.hstack(( x_init_vec, vx_init_vec, ax_init_vec ))
		b_eq_y = jnp.hstack(( y_init_vec, vy_init_vec, ay_init_vec, jnp.zeros((self.num_batch, 1)) ))

		return b_eq_x, b_eq_y

	@partial(jit, static_argnums=(0,))
	def initial_alpha_d_obs(self, x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_obs, y_obs, lamda_x, lamda_y):

		wc_alpha_temp = (x_guess-x_obs[:,jnp.newaxis])
		ws_alpha_temp = (y_guess-y_obs[:,jnp.newaxis])

		wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
		ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

		wc_alpha = wc_alpha.reshape(self.num_batch, self.num*self.num_obs)
		ws_alpha = ws_alpha.reshape(self.num_batch, self.num*self.num_obs)

		alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
		c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
		c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

		d_temp = c2_d/c1_d
		d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.num*self.num_obs   )), d_temp   )
		

		################# velocity terms

		wc_alpha_vx = xdot_guess
		ws_alpha_vy = ydot_guess
		alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
		
		
		c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
		
		d_temp_v = c2_d_v/c1_d_v
		
		d_v = jnp.clip(d_temp_v, self.v_min, self.v_max )
		
		################# acceleration terms

		wc_alpha_ax = xddot_guess
		ws_alpha_ay = yddot_guess
		alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
		
	
		c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

		d_temp_a = c2_d_a/c1_d_a
		d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), self.a_max  )

		
		#########################################33
		res_ax_vec = xddot_guess-d_a*jnp.cos(alpha_a)
		res_ay_vec = yddot_guess-d_a*jnp.sin(alpha_a)
		
		res_vx_vec = xdot_guess-d_v*jnp.cos(alpha_v)
		res_vy_vec = ydot_guess-d_v*jnp.sin(alpha_v)

		res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
		res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

			
		res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
		res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
		res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

		
		lamda_x = lamda_x-self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
		lamda_y = lamda_y-self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T
	
		return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v

	@partial(jit, static_argnums=(0,))	
	def compute_x(self, lamda_x, lamda_y, b_eq_x, b_eq_y, alpha_a, d_a, alpha_v, d_v, x_obs, y_obs, alpha_obs, d_obs, c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub):

		
		
		b_lane = jnp.hstack(( y_lane_ub*jnp.ones(( self.num_batch, self.num  )), -y_lane_lb*jnp.ones(( self.num_batch, self.num  ))     ))

		b_lane_aug = b_lane-s_lane
		
		b_ax_ineq = d_a*jnp.cos(alpha_a)
		b_ay_ineq = d_a*jnp.sin(alpha_a)

		b_vx_ineq = d_v*jnp.cos(alpha_v)
		b_vy_ineq = d_v*jnp.sin(alpha_v)


		temp_x_obs = d_obs*jnp.cos(alpha_obs)*self.a_obs
		b_obs_x = x_obs.reshape(self.num*self.num_obs)+temp_x_obs
		 
		temp_y_obs = d_obs*jnp.sin(alpha_obs)*self.b_obs
		b_obs_y = y_obs.reshape(self.num*self.num_obs)+temp_y_obs

		cost_x = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)
		cost_y = self.rho_projection*jnp.dot(self.A_projection.T, self.A_projection)+self.rho_obs*jnp.dot(self.A_obs.T, self.A_obs)+self.rho_ineq*jnp.dot(self.A_acc.T, self.A_acc)+self.rho_ineq*jnp.dot(self.A_vel.T, self.A_vel)+self.rho_lane*jnp.dot(self.A_lane.T, self.A_lane)

		cost_mat_x = jnp.vstack((  jnp.hstack(( cost_x, self.A_eq_x.T )), jnp.hstack(( self.A_eq_x, jnp.zeros(( jnp.shape(self.A_eq_x)[0], jnp.shape(self.A_eq_x)[0] )) )) ))
		
		cost_mat_y = jnp.vstack((  jnp.hstack(( cost_y, self.A_eq_y.T )), jnp.hstack(( self.A_eq_y, jnp.zeros(( jnp.shape(self.A_eq_y)[0], jnp.shape(self.A_eq_y)[0] )) )) ))
		
		lincost_x = -lamda_x-self.rho_projection*jnp.dot(self.A_projection.T, c_x_bar.T).T-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_x.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ax_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vx_ineq.T).T
		lincost_y = -lamda_y-self.rho_projection*jnp.dot(self.A_projection.T, c_y_bar.T).T-self.rho_obs*jnp.dot(self.A_obs.T, b_obs_y.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, b_ay_ineq.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, b_vy_ineq.T).T-self.rho_lane*jnp.dot(self.A_lane.T, b_lane_aug.T).T
	
		sol_x = jnp.linalg.solve(cost_mat_x, jnp.hstack(( -lincost_x, b_eq_x )).T).T
		sol_y = jnp.linalg.solve(cost_mat_y, jnp.hstack(( -lincost_y, b_eq_y )).T).T

		primal_sol_x = sol_x[:,0:self.nvar]
		primal_sol_y = sol_y[:,0:self.nvar]

		x = jnp.dot(self.P_jax, primal_sol_x.T).T
		xdot = jnp.dot(self.Pdot_jax, primal_sol_x.T).T
		xddot = jnp.dot(self.Pddot_jax, primal_sol_x.T).T


		y = jnp.dot(self.P_jax, primal_sol_y.T).T
		ydot = jnp.dot(self.Pdot_jax, primal_sol_y.T).T
		yddot = jnp.dot(self.Pddot_jax, primal_sol_y.T).T

		s_lane = jnp.maximum( jnp.zeros(( self.num_batch, 2*self.num )), -jnp.dot(self.A_lane, primal_sol_y.T).T+b_lane  )

		res_lane_vec = jnp.dot(self.A_lane, primal_sol_y.T).T-b_lane+s_lane


		return primal_sol_x, primal_sol_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane

	@partial(jit, static_argnums=(0,))	
	def compute_alph_d(self, x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y, alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev, x_obs, y_obs, y_lane_lb, y_lane_ub, res_lane_vec):
		
		wc_alpha_temp = (x-x_obs[:,jnp.newaxis])
		ws_alpha_temp = (y-y_obs[:,jnp.newaxis])

		wc_alpha = wc_alpha_temp.transpose(1, 0, 2)
		ws_alpha = ws_alpha_temp.transpose(1, 0, 2)

		wc_alpha = wc_alpha.reshape(self.num_batch, self.num*self.num_obs)
		ws_alpha = ws_alpha.reshape(self.num_batch, self.num*self.num_obs)

		alpha_obs = jnp.arctan2( ws_alpha*self.a_obs, wc_alpha*self.b_obs)
		c1_d = 1.0*self.rho_obs*(self.a_obs**2*jnp.cos(alpha_obs)**2 + self.b_obs**2*jnp.sin(alpha_obs)**2 )
		c2_d = 1.0*self.rho_obs*(self.a_obs*wc_alpha*jnp.cos(alpha_obs) + self.b_obs*ws_alpha*jnp.sin(alpha_obs)  )

		d_temp = c2_d/c1_d
		d_obs = jnp.maximum(jnp.ones((self.num_batch,  self.num*self.num_obs   )), d_temp   )
		

		################# velocity terms

		wc_alpha_vx = xdot
		ws_alpha_vy = ydot
		alpha_v = jnp.unwrap(jnp.arctan2( ws_alpha_vy, wc_alpha_vx))
		
		
		kappa_bound_d_v = jnp.sqrt(d_a_prev*jnp.abs(jnp.sin(alpha_a_prev-alpha_v))/self.kappa_max)

		v_min_aug = jnp.maximum( kappa_bound_d_v, self.v_min*jnp.ones(( self.num_batch, self.num  ))  )

		c1_d_v = 1.0*self.rho_ineq*(jnp.cos(alpha_v)**2 + jnp.sin(alpha_v)**2 )
		c2_d_v = 1.0*self.rho_ineq*(wc_alpha_vx*jnp.cos(alpha_v) + ws_alpha_vy*jnp.sin(alpha_v)  )
		
		d_temp_v = c2_d_v/c1_d_v
		
		d_v = jnp.clip(d_temp_v, v_min_aug, self.v_max )
		
		################# acceleration terms

		wc_alpha_ax = xddot
		ws_alpha_ay = yddot
		alpha_a = jnp.unwrap(jnp.arctan2( ws_alpha_ay, wc_alpha_ax))
		
	
		c1_d_a = 1.0*self.rho_ineq*(jnp.cos(alpha_a)**2 + jnp.sin(alpha_a)**2 )
		c2_d_a = 1.0*self.rho_ineq*(wc_alpha_ax*jnp.cos(alpha_a) + ws_alpha_ay*jnp.sin(alpha_a)  )

		kappa_bound_d_a = (self.kappa_max*d_v**2)/jnp.abs(jnp.sin(alpha_a-alpha_v))
		a_max_aug = jnp.minimum( self.a_max*jnp.ones((self.num_batch, self.num)), kappa_bound_d_a )

		d_temp_a = c2_d_a/c1_d_a
		d_a = jnp.clip(d_temp_a, jnp.zeros((self.num_batch, self.num)), a_max_aug  )

		
		#######################################################

		#########################################33
		res_ax_vec = xddot-d_a*jnp.cos(alpha_a)
		res_ay_vec = yddot-d_a*jnp.sin(alpha_a)
		
		res_vx_vec = xdot-d_v*jnp.cos(alpha_v)
		res_vy_vec = ydot-d_v*jnp.sin(alpha_v)

		res_x_obs_vec = wc_alpha-self.a_obs*d_obs*jnp.cos(alpha_obs)
		res_y_obs_vec = ws_alpha-self.b_obs*d_obs*jnp.sin(alpha_obs)

			
		res_vel_vec = jnp.hstack(( res_vx_vec,  res_vy_vec  ))
		res_acc_vec = jnp.hstack(( res_ax_vec,  res_ay_vec  ))
		res_obs_vec = jnp.hstack(( res_x_obs_vec, res_y_obs_vec  ))

		res_norm_batch = jnp.linalg.norm(res_obs_vec, axis =1)+jnp.linalg.norm(res_acc_vec, axis =1)+jnp.linalg.norm(res_vel_vec, axis =1)+jnp.linalg.norm(res_lane_vec, axis = 1)

		lamda_x = lamda_x-self.rho_obs*jnp.dot(self.A_obs.T, res_x_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ax_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vx_vec.T).T
		lamda_y = lamda_y-self.rho_obs*jnp.dot(self.A_obs.T, res_y_obs_vec.T).T-self.rho_ineq*jnp.dot(self.A_acc.T, res_ay_vec.T).T-self.rho_ineq*jnp.dot(self.A_vel.T, res_vy_vec.T).T-self.rho_lane*jnp.dot(self.A_lane.T, res_lane_vec.T).T
	
		return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v

	@partial(jit, static_argnums=(0, ))	
	def compute_projection(self, initial_state, lamda_x, lamda_y, x_obs, y_obs, c_x_bar, c_y_bar, y_lane_lb, y_lane_ub, alpha_a, d_a, alpha_v, d_v, s_lane):

		x_init, y_init, vx_init, vy_init, ax_init, ay_init = initial_state

		b_eq_x, b_eq_y = self.compute_boundary_vec(x_init, vx_init, ax_init, y_init, vy_init, ay_init)

		# alpha_obs, d_obs = self.initial_alpha_d_obs(x_guess, y_guess, x_obs, y_obs)


		x_guess = jnp.dot(self.P_jax, c_x_bar.T).T 
		y_guess = jnp.dot(self.P_jax, c_y_bar.T).T 
		

		xdot_guess = jnp.dot(self.Pdot_jax, c_x_bar.T).T 
		ydot_guess = jnp.dot(self.Pdot_jax, c_y_bar.T).T 


		xddot_guess = jnp.dot(self.Pddot_jax, c_x_bar.T).T 
		yddot_guess = jnp.dot(self.Pddot_jax, c_y_bar.T).T 
		

		alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v = self.initial_alpha_d_obs(x_guess, y_guess, xdot_guess, ydot_guess, xddot_guess, yddot_guess, x_obs, y_obs, lamda_x, lamda_y)

		for i in range(0, self.maxiter):
		
			alpha_a_prev = alpha_a
			alpha_v_prev = alpha_v 
			d_v_prev = d_v
			d_a_prev = d_a

			c_x, c_y, x, y, xdot, ydot, xddot, yddot, res_lane_vec, s_lane = self.compute_x(lamda_x, lamda_y, b_eq_x, b_eq_y, alpha_a, d_a, alpha_v, d_v, x_obs, y_obs, alpha_obs, d_obs, c_x_bar, c_y_bar, s_lane, y_lane_lb, y_lane_ub)

			alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v = self.compute_alph_d(x, y, xdot, ydot, xddot, yddot, lamda_x, lamda_y, alpha_v_prev, alpha_a_prev, d_v_prev, d_a_prev, x_obs, y_obs, y_lane_lb, y_lane_ub, res_lane_vec)
		
			curvature = d_a*jnp.sin(alpha_a-alpha_v)/(d_v**2)
			steering = jnp.arctan(curvature*self.wheel_base  )

		return 	c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v, curvature, steering, s_lane

	@partial(jit, static_argnums=(0, ))
	def compute_cost(self, y_ellite_projection, res_ellite_projection, xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, y_lane_lb, y_lane_ub, v_des, steering_ellite_projection, d_v_ellite_projection):
		
		temp_vx = xddot_ellite_projection-self.k_p*(xdot_ellite_projection-v_des)#-self.k_d*xddot_ellite_projection
		
		temp_vy = yddot_ellite_projection-self.k_p_1*(ydot_ellite_projection)#-self.k_d_1*yddot_ellite_projection

		cost_centerline_penalty_lb = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), -y_ellite_projection+(y_lane_lb)), axis = 1)
		cost_centerline_penalty_ub = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), y_ellite_projection-y_lane_ub), axis = 1)
		
		cost_steering = jnp.linalg.norm(steering_ellite_projection, axis = 1)
		steering_vel = jnp.diff(steering_ellite_projection, axis = 1)
		cost_steering_vel = jnp.linalg.norm(steering_vel, axis = 1)

		temp_steering = steering_vel-self.k_p*(steering_ellite_projection[:, 1:])

		heading_angle = jnp.arctan2(ydot_ellite_projection, xdot_ellite_projection)
		heading_penalty = jnp.linalg.norm(jnp.maximum(jnp.zeros(( self.ellite_num_projection, self.num  )), jnp.abs(heading_angle)-10*jnp.pi/180   ), axis = 1)
		dist_cost = jnp.linalg.norm(d_v_ellite_projection, axis = 1)
		
		# cost_batch = res_ellite_projection+(cost_centerline_penalty_lb+cost_centerline_penalty_ub)+0.01*jnp.linalg.norm(temp_vx, axis = 1)+0.001*jnp.linalg.norm(temp_vy, axis = 1)+cost_steering+cost_steering_vel+10*heading_penalty+0.1*jnp.linalg.norm(y_ellite_projection-y_lane_ub)

		# cost_batch = res_ellite_projection+0.001*jnp.linalg.norm(temp_vx, axis = 1)+jnp.linalg.norm(temp_steering, axis = 1)+0.001*cost_steering+0.001*cost_steering_vel+0.001*heading_penalty+0.001*jnp.linalg.norm(y_ellite_projection-(y_lane_ub+y_lane_lb)/2, axis = 1  )

		v = jnp.sqrt(xdot_ellite_projection**2+ydot_ellite_projection**2)

		cost_batch = res_ellite_projection+0.01*jnp.linalg.norm(xdot_ellite_projection-v_des, axis = 1)+0.01*cost_steering+0.01*cost_steering_vel+heading_penalty#+0.001*jnp.linalg.norm(y_ellite_projection-(y_lane_ub+y_lane_lb)/2, axis = 1  )
		
		
		cost_batch_scale = cost_batch

		return cost_batch_scale

	@partial(jit, static_argnums=(0, ))
	def compute_ellite_samples(self, cost_batch, neural_output_projection):

		idx_ellite = jnp.argsort(cost_batch)

		# c_x_ellite = c_x_ellite_projection[idx_ellite[0:self.ellite_num]]
		# c_y_ellite = c_y_ellite_projection[idx_ellite[0:self.ellite_num]]

		neural_output_ellite = neural_output_projection[idx_ellite[0:self.ellite_num]]

		return neural_output_ellite, idx_ellite

	@partial(jit, static_argnums=(0,))
	def comp_prod(self, diffs, d ):
		term_1 = jnp.expand_dims(diffs, axis = 1)
		term_2 = jnp.expand_dims(diffs, axis = 0)
		prods = d * jnp.outer(term_1,term_2)
		# prods = d*jnp.outer(diffs,diffs)
		return prods	

	@partial(jit, static_argnums=(0, ))
	def compute_shifted_samples(self, key, neural_output_ellite, cost_batch, idx_ellite, mean_param_prev, cov_param_prev):

		
		# temp = jnp.maximum(1, (15.5/((i+1))  ))

		mean_param = jnp.mean(neural_output_ellite, axis = 0)
		cov_param = jnp.cov(neural_output_ellite.T)+0.01*jnp.identity(8)

		# c_mean = (1-self.alpha_mean)*c_mean_prev+self.alpha_mean*c_mean
		# c_cov = (1-self.alpha_cov)*c_cov_prev+self.alpha_cov*c_cov

		cost_batch_temp = cost_batch[idx_ellite[0:self.ellite_num]]

		######################################

		# w = cost_batch_temp
		# w_min = jnp.min(cost_batch_temp)
		# w = jnp.exp(-(1/self.lamda) * (w - w_min ) )
		# sum_w = jnp.sum(w, axis = 0)
		# mean_param = (1-self.alpha_mean)*mean_param_prev + self.alpha_mean*(jnp.sum( (neural_output_ellite * w[:,jnp.newaxis]) , axis= 0)/ sum_w)
		# diffs = (neural_output_ellite - mean_param)
		# prod_result = self.vec_product(diffs, w)
		# cov_param = (1-self.alpha_cov)*cov_param_prev + self.alpha_cov*(jnp.sum( prod_result , axis = 0)/jnp.sum(w, axis = 0)) + 0.01*jnp.identity(8)


		param_samples = jax.random.multivariate_normal(key, mean_param, cov_param, (self.num_batch-self.ellite_num, ))

		v_des_1 = param_samples[:, 0]
		v_des_2 = param_samples[:, 1]
		v_des_3 = param_samples[:, 2]
		v_des_4 = param_samples[:, 3]
			
		
		y_des_1 = param_samples[:, 4]
		y_des_2 = param_samples[:, 5]
		y_des_3 = param_samples[:, 6]
		y_des_4 = param_samples[:, 7]
		
		# y_des_1 = jnp.clip(y_des_1, y_lane_lb*jnp.ones(self.num_batch), y_lane_ub*jnp.ones(self.num_batch)   )
		# y_des_2 = jnp.clip(y_des_2, y_lane_lb*jnp.ones(self.num_batch), y_lane_ub*jnp.ones(self.num_batch)   )
		
		# y_des_3 = jnp.clip(y_des_3, y_lane_lb*jnp.ones(self.num_batch), y_lane_ub*jnp.ones(self.num_batch)   )
		# y_des_4 = jnp.clip(y_des_4, y_lane_lb*jnp.ones(self.num_batch), y_lane_ub*jnp.ones(self.num_batch)   )
		
		v_des_1 = jnp.clip(v_des_1, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
		v_des_2 = jnp.clip(v_des_2, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
		
		v_des_3 = jnp.clip(v_des_3, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
		
		v_des_4 = jnp.clip(v_des_4, self.v_min*jnp.ones(self.num_batch-self.ellite_num), self.v_max*jnp.ones(self.num_batch-self.ellite_num)   )
		

		neural_output_shift = jnp.vstack(( v_des_1, v_des_2, v_des_3, v_des_4, y_des_1, y_des_2, y_des_3, y_des_4   )).T

		neural_output_batch = jnp.vstack(( neural_output_ellite, neural_output_shift  ))

		return mean_param, cov_param, neural_output_batch, cost_batch_temp

	@partial(jit, static_argnums=(0, ))	
	def compute_controls(self, c_x_best, c_y_best):

		xdot_best = jnp.dot(self.Pdot_jax_up, c_x_best)
		ydot_best = jnp.dot(self.Pdot_jax_up, c_y_best)

		xddot_best = jnp.dot(self.Pddot_jax_up, c_x_best)
		yddot_best = jnp.dot(self.Pddot_jax_up, c_y_best)

		curvature_best = (yddot_best*xdot_best-ydot_best*xddot_best)/((xdot_best**2+ydot_best**2)**(1.5)) 
		steer_best = jnp.arctan(curvature_best*self.wheel_base  )

		v_best = jnp.sqrt(xdot_best**2+ydot_best**2)
		a_best = jnp.diff(v_best, axis = 0)/self.t_up

		return a_best, steer_best

	@partial(jit, static_argnums=(0, ))	
	def compute_cem(self, initial_state, lamda_x, lamda_y, x_obs, y_obs, y_lane_lb, y_lane_ub,  alpha_a, d_a, alpha_v, d_v, v_des, s_lane, mean_param, cov_param):

		# res = []
		# res_2 = []

		# alpha_obs, d_obs = self.initial_alpha_d_obs(x_guess, y_guess, x_obs, y_obs)
		
		neural_output_batch = self.sampling_param(y_lane_lb, y_lane_ub, mean_param, cov_param, v_des)

		# neural_output_batch = jnp.vstack(( neural_output_batch, neural_output_warmstart  ))


		for i in range(0, self.maxiter_cem):

			mean_param_prev = mean_param
			cov_param_prev = cov_param


			
			c_x_bar, c_y_bar = self.compute_x_guess(initial_state, neural_output_batch)

			c_x, c_y, x, y, xdot, ydot, xddot, yddot, alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, res_norm_batch, alpha_v, d_v, curvature, steering, s_lane = self.compute_projection(initial_state, lamda_x, lamda_y, x_obs, y_obs, c_x_bar, c_y_bar, y_lane_lb, y_lane_ub, alpha_a, d_a, alpha_v, d_v, s_lane)
				
			idx_ellite_projection = jnp.argsort(res_norm_batch)

			x_ellite_projection = x[idx_ellite_projection[0:self.ellite_num_projection]]
			y_ellite_projection = y[idx_ellite_projection[0:self.ellite_num_projection]]

			xdot_ellite_projection = xdot[idx_ellite_projection[0:self.ellite_num_projection]]
			ydot_ellite_projection = ydot[idx_ellite_projection[0:self.ellite_num_projection]]
			
			xddot_ellite_projection = xddot[idx_ellite_projection[0:self.ellite_num_projection]]
			yddot_ellite_projection = yddot[idx_ellite_projection[0:self.ellite_num_projection]]

			steering_ellite_projection = steering[idx_ellite_projection[0:self.ellite_num_projection]]

			c_x_ellite_projection = c_x[idx_ellite_projection[0:self.ellite_num_projection]]
			c_y_ellite_projection = c_y[idx_ellite_projection[0:self.ellite_num_projection]]
			d_v_ellite_projection = d_v[idx_ellite_projection[0:self.ellite_num_projection]]
			res_ellite_projection = res_norm_batch[idx_ellite_projection[0:self.ellite_num_projection]]

			neural_output_projection = neural_output_batch[idx_ellite_projection[0:self.ellite_num_projection]]
	
			cost_batch = self.compute_cost(y_ellite_projection, res_ellite_projection, xdot_ellite_projection, ydot_ellite_projection, xddot_ellite_projection, yddot_ellite_projection, y_lane_lb, y_lane_ub, v_des, steering_ellite_projection, d_v_ellite_projection)
						
			idx_min = jnp.argmin(cost_batch)

		c_x_best = c_x_ellite_projection[idx_min]
		c_y_best = c_y_ellite_projection[idx_min]
				
		return 	c_x_best, c_y_best