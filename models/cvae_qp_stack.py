import numpy as np
import torch
import torch.nn as nn 

# Reproducibility
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.set_default_dtype(torch.float32)

# GPU Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prevents NaN by torch.log(0)
def torch_log(x):
	return torch.log(torch.clamp(x, min = 1e-10))

# Encoder
class Encoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Encoder, self).__init__()
				
		# Encoder Architecture
		self.encoder = nn.Sequential(
			nn.Linear(inp_dim + out_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(), 
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU()
		)
		
		# Mean and Variance
		self.mu = nn.Linear(256, z_dim)
		self.var = nn.Linear(256, z_dim)
		
		self.softplus = nn.Softplus()
		
	def forward(self, x):
		out = self.encoder(x)
		mu = self.mu(out)
		var = self.var(out)
		return mu, self.softplus(var)
	
# Decoder
class Decoder(nn.Module):
	def __init__(self, inp_dim, out_dim, hidden_dim, z_dim):
		super(Decoder, self).__init__()
		
		# Decoder Architecture
		self.decoder = nn.Sequential(
			nn.Linear(z_dim + inp_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, hidden_dim),
			nn.BatchNorm1d(hidden_dim),
			nn.ReLU(),
			
			nn.Linear(hidden_dim, 256),
			nn.BatchNorm1d(256),
			nn.ReLU(),
			
			nn.Linear(256, out_dim),
		)
	
	def forward(self, x):
		out = self.decoder(x)
		return out

class Beta_cVAE(nn.Module):
	def __init__(self, P, Pdot, Pddot, encoder, decoder, num_batch, inp_mean, inp_std):
		super(Beta_cVAE, self).__init__()
		
		# Encoder & Decoder
		self.encoder = encoder
		self.decoder = decoder
		
		# Normalizing Constants
		self.inp_mean = inp_mean
		self.inp_std = inp_std

		# P Matrices
		self.P = P.to(device)
		self.Pdot = Pdot.to(device)
		self.Pddot = Pddot.to(device)
		
		# Bernstein P
		self.P1 = self.P[0:25, :]
		self.P2 = self.P[25:50, :]
		self.P3 = self.P[50:75, :]
		self.P4 = self.P[75:100, :]

		self.Pdot1 = self.Pdot[0:25, :]
		self.Pdot2 = self.Pdot[25:50, :]
		self.Pdot3 = self.Pdot[50:75, :]
		self.Pdot4 = self.Pdot[75:100, :]
			
		self.Pddot1 = self.Pddot[0:25, :]
		self.Pddot2 = self.Pddot[25:50, :]
		self.Pddot3 = self.Pddot[50:75, :]
		self.Pddot4 = self.Pddot[75:100, :]

		self.A_eq_x = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0]])
		self.A_eq_y = torch.vstack([self.P[0], self.Pdot[0], self.Pddot[0], self.Pdot[-1]])

		# K constants
		self.k_p = torch.tensor(20.0, device=device)
		self.k_d = 2.0 * torch.sqrt(self.k_p)

		self.k_p_v = torch.tensor(20.0, device=device)
		self.k_d_v = 2.0 * torch.sqrt(self.k_p_v)  
		self.kappa_max = torch.tensor(0.23, device=device)
		
		# No. of Variables
		self.nvar = 11
		self.num_batch = num_batch
		self.a_obs = 8.0
		self.b_obs = 4.2
		
		# Parameters
		self.rho_v = 1.0  
		self.rho_projection = 1.0
		self.rho_lane = 100.0
		self.rho_obs = 100
		self.rho_offset = 1
		self.rho_ineq = 100
		t_fin = 15.0
		self.num = 100
		self.tot_time = torch.linspace(0, t_fin, self.num, device=device)
		self.num_obs = 10
		self.num_mean_update = 8
		self.t = t_fin / self.num
		self.t_target = (self.num_mean_update - 1) * self.t
		self.v_min = 0.1
		self.v_max = 30
		self.a_max = 8.0
		self.wheel_base = 2.5
		self.steer_max = 0.6
		self.v_des = 20.0
		self.num_partial = 25
		self.A_obs = torch.tile(self.P, (self.num_obs, 1))
		self.A_lane = torch.vstack((self.P, -self.P))	
		self.A_vel = self.Pdot
		self.A_acc = self.Pddot
		self.A_projection = torch.eye(self.nvar, device=device)
		self.maxiter = 20 # 20
		
		# Smoothness
		self.weight_smoothness = 1.0 
		self.cost_smoothness = self.weight_smoothness * torch.mm(self.Pddot.T, self.Pddot)
		self.weight_aug = 1.0
		self.vel_scale = 1e-3 # 1e-3
		
  		# RCL Loss
		self.rcl_loss = nn.MSELoss()
	
	# Inverse Matrices
	def compute_mat_inv_1(self):
		
		A_pd_1 = self.Pddot1 - self.k_p * self.P1 - self.k_d * self.Pdot1
		A_pd_2 = self.Pddot2 - self.k_p * self.P2 - self.k_d * self.Pdot2
		A_pd_3 = self.Pddot3 - self.k_p * self.P3 - self.k_d * self.Pdot3
		A_pd_4 = self.Pddot4 - self.k_p * self.P4 - self.k_d * self.Pdot4
		A_vd_1 = self.Pddot1 - self.k_p_v * self.Pdot1
		A_vd_2 = self.Pddot2 - self.k_p_v * self.Pdot2
		A_vd_3 = self.Pddot3 - self.k_p_v * self.Pdot3
		A_vd_4 = self.Pddot4 - self.k_p_v * self.Pdot4

		cost_x = self.cost_smoothness + self.rho_v * torch.mm(A_vd_1.T, A_vd_1) + \
      			 self.rho_v * torch.mm(A_vd_2.T, A_vd_2) + self.rho_v * torch.mm(A_vd_3.T, A_vd_3) + \
              	 self.rho_v * torch.mm(A_vd_4.T, A_vd_4)
                
		cost_y = self.cost_smoothness + self.rho_offset * torch.mm(A_pd_1.T, A_pd_1) + \
      			 self.rho_offset * torch.mm(A_pd_2.T, A_pd_2) + self.rho_offset * torch.mm(A_pd_3.T, A_pd_3) + \
              	 self.rho_offset * torch.mm(A_pd_4.T, A_pd_4)
                
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x.T]), torch.hstack([self.A_eq_x, torch.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y.T]), torch.hstack([self.A_eq_y, torch.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device)])])

		cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
		cost_mat_inv_y = torch.linalg.inv(cost_mat_y)
		
		return cost_mat_inv_x, cost_mat_inv_y

	def compute_mat_inv_2(self):
     
		cost_x = self.rho_projection * torch.mm(self.A_projection.T, self.A_projection) + \
      			 self.rho_obs * torch.mm(self.A_obs.T, self.A_obs) + \
              	 self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc) + \
                 self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel)
                 
		cost_y = self.rho_projection * torch.mm(self.A_projection.T, self.A_projection) + \
            	 self.rho_obs * torch.mm(self.A_obs.T, self.A_obs) + \
                 self.rho_ineq * torch.mm(self.A_acc.T, self.A_acc) + \
                 self.rho_ineq * torch.mm(self.A_vel.T, self.A_vel) + \
                 self.rho_lane * torch.mm(self.A_lane.T, self.A_lane)
        
		cost_mat_x = torch.vstack([torch.hstack([cost_x, self.A_eq_x.T]), torch.hstack([self.A_eq_x, torch.zeros((self.A_eq_x.shape[0], self.A_eq_x.shape[0]), device=device)])])
		cost_mat_y = torch.vstack([torch.hstack([cost_y, self.A_eq_y.T]), torch.hstack([self.A_eq_y, torch.zeros((self.A_eq_y.shape[0], self.A_eq_y.shape[0]), device=device)])])

		cost_mat_inv_x = torch.linalg.inv(cost_mat_x)
		cost_mat_inv_y = torch.linalg.inv(cost_mat_y)
  
		return cost_mat_inv_x, cost_mat_inv_y

	# Boundary Vectors
	def compute_boundary(self, initial_state_ego):
	 
		x_init_vec = torch.zeros([self.num_batch, 1], device=device) 
		y_init_vec = torch.zeros([self.num_batch, 1], device=device) 
  
		vx_init_vec = initial_state_ego[:, 2].reshape(self.num_batch, 1)
		vy_init_vec = initial_state_ego[:, 3].reshape(self.num_batch, 1)

		ax_init_vec = torch.zeros([self.num_batch, 1], device=device)
		ay_init_vec = torch.zeros([self.num_batch, 1], device=device)

		b_eq_x = torch.hstack([x_init_vec, vx_init_vec, ax_init_vec])
		b_eq_y = torch.hstack([y_init_vec, vy_init_vec, ay_init_vec, torch.zeros((self.num_batch, 1), device=device)])
	
		return b_eq_x, b_eq_y

	def compute_obs_trajectories(self, inp):
     
		# Obstacle coordinates & Velocity 
		x_obs = inp[:, 5::5]
		y_obs = inp[:, 6::5]
		vx_obs = inp[:, 7::5]
		vy_obs = inp[:, 8::5]

		# Batch Obstacle Trajectory Prediction
		x_obs_inp_trans = x_obs.reshape(self.num_batch, 1, self.num_obs)
		y_obs_inp_trans = y_obs.reshape(self.num_batch, 1, self.num_obs)

		vx_obs_inp_trans = vx_obs.reshape(self.num_batch, 1, self.num_obs)
		vy_obs_inp_trans = vy_obs.reshape(self.num_batch, 1, self.num_obs)

		x_obs_traj = x_obs_inp_trans + vx_obs_inp_trans * self.tot_time.unsqueeze(1)
		y_obs_traj = y_obs_inp_trans + vy_obs_inp_trans * self.tot_time.unsqueeze(1)

		x_obs_traj = x_obs_traj.permute(0, 2, 1)
		y_obs_traj = y_obs_traj.permute(0, 2, 1)

		x_obs_traj = x_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
		y_obs_traj = y_obs_traj.reshape(self.num_batch, self.num_obs * self.num)
  
		return x_obs_traj, y_obs_traj

	# Solve Function
	def qp_layer_1(self, initial_state_ego, neural_output_batch):
		
		# Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary(initial_state_ego) 
  
		# Inverse Matrices
		cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_1()

		# Predicted Behavioral Params
		v_des_1 = neural_output_batch[:, 0]
		v_des_2 = neural_output_batch[:, 1]
		v_des_3 = neural_output_batch[:, 2]
		v_des_4 = neural_output_batch[:, 3]

		y_des_1 = neural_output_batch[:, 4]
		y_des_2 = neural_output_batch[:, 5]
		y_des_3 = neural_output_batch[:, 6]
		y_des_4 = neural_output_batch[:, 7]

		# A & b Matrices
		A_pd_1 = self.Pddot1 - self.k_p * self.P1 - self.k_d * self.Pdot1
		b_pd_1 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_1)[:, None]

		A_pd_2 = self.Pddot2 - self.k_p * self.P2 - self.k_d * self.Pdot2
		b_pd_2 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_2)[:, None]
			
		A_pd_3 = self.Pddot3 - self.k_p * self.P3 - self.k_d * self.Pdot3
		b_pd_3 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_3)[:, None]

		A_pd_4 = self.Pddot4 - self.k_p * self.P4 - self.k_d * self.Pdot4
		b_pd_4 = -self.k_p * torch.ones((self.num_batch, self.num_partial), device=device) * (y_des_4)[:, None]

		A_vd_1 = self.Pddot1 - self.k_p_v * self.Pdot1 #- self.k_d_v * self.Pddot1
		b_vd_1 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_1)[:, None]

		A_vd_2 = self.Pddot2 - self.k_p_v * self.Pdot2 #- self.k_d_v * self.Pddot2
		b_vd_2 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_2)[:, None]

		A_vd_3 = self.Pddot3 - self.k_p_v * self.Pdot3 #- self.k_d_v * self.Pddot3
		b_vd_3 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_3)[:, None]

		A_vd_4 = self.Pddot4 - self.k_p_v * self.Pdot4 #- self.k_d_v * self.Pddot4
		b_vd_4 = -self.k_p_v * torch.ones((self.num_batch, self.num_partial), device=device) * (v_des_4)[:, None]
  
		lincost_x = -self.rho_v * torch.mm(A_vd_1.T, b_vd_1.T).T - self.rho_v * torch.mm(A_vd_2.T, b_vd_2.T).T - self.rho_v * torch.mm(A_vd_3.T, b_vd_3.T).T - self.rho_v * torch.mm(A_vd_4.T, b_vd_4.T).T
		lincost_y = -self.rho_offset * torch.mm(A_pd_1.T, b_pd_1.T).T - self.rho_offset * torch.mm(A_pd_2.T, b_pd_2.T).T - self.rho_offset * torch.mm(A_pd_3.T, b_pd_3.T).T - self.rho_offset * torch.mm(A_pd_4.T, b_pd_4.T).T

		sol_x = torch.mm(cost_mat_inv_x, torch.hstack([-lincost_x, b_eq_x]).T).T
		sol_y = torch.mm(cost_mat_inv_y, torch.hstack([-lincost_y, b_eq_y]).T).T
		
		c_x = sol_x[:, 0:self.nvar]
		c_y = sol_y[:, 0:self.nvar]

		# Solution
		y = torch.hstack([c_x, c_y])

		return y

	def compute_alph_d(self, primal_sol, x_obs_traj, y_obs_traj, y_ub, y_lb, lamda_x, lamda_y):
     
		primal_sol_x = primal_sol[:, 0:self.nvar]
		primal_sol_y = primal_sol[:, self.nvar:2 * self.nvar]

		x = torch.mm(self.P, primal_sol_x.T).T
		xdot = torch.mm(self.Pdot, primal_sol_x.T).T 
		xddot = torch.mm(self.Pddot, primal_sol_x.T).T
  
		y = torch.mm(self.P, primal_sol_y.T).T
		ydot = torch.mm(self.Pdot, primal_sol_y.T).T
		yddot = torch.mm(self.Pddot, primal_sol_y.T).T

		x_extend = torch.tile(x, (1, self.num_obs))
		y_extend = torch.tile(y, (1, self.num_obs))

		wc_alpha = (x_extend - x_obs_traj)
		ws_alpha = (y_extend - y_obs_traj)

		wc_alpha = wc_alpha.reshape(self.num_batch, self.num * self.num_obs)
		ws_alpha = ws_alpha.reshape(self.num_batch, self.num * self.num_obs)
  
		alpha_obs = torch.arctan2(ws_alpha * self.a_obs, wc_alpha * self.b_obs)
		c1_d = 1.0 * self.rho_obs*(self.a_obs**2 * torch.cos(alpha_obs)**2 + self.b_obs**2 * torch.sin(alpha_obs)**2)
		c2_d = 1.0 * self.rho_obs*(self.a_obs * wc_alpha * torch.cos(alpha_obs) + self.b_obs * ws_alpha * torch.sin(alpha_obs))
  
		d_temp = c2_d/c1_d
		d_obs = torch.maximum(torch.ones((self.num_batch, self.num * self.num_obs), device=device), d_temp)
  
		wc_alpha_vx = xdot
		ws_alpha_vy = ydot
		alpha_v = torch.arctan2( ws_alpha_vy, wc_alpha_vx)
		
		c1_d_v = 1.0 * self.rho_ineq * (torch.cos(alpha_v)**2 + torch.sin(alpha_v)**2)
		c2_d_v = 1.0 * self.rho_ineq * (wc_alpha_vx * torch.cos(alpha_v) + ws_alpha_vy * torch.sin(alpha_v))
		
		d_temp_v = c2_d_v/c1_d_v
		d_v = torch.clip(d_temp_v, torch.tensor(self.v_min).to(device), torch.tensor(self.v_max).to(device))

		wc_alpha_ax = xddot
		ws_alpha_ay = yddot
		alpha_a = torch.arctan2( ws_alpha_ay, wc_alpha_ax)
		
		c1_d_a = 1.0 * self.rho_ineq * (torch.cos(alpha_a)**2 + torch.sin(alpha_a)**2)
		c2_d_a = 1.0 * self.rho_ineq * (wc_alpha_ax * torch.cos(alpha_a) + ws_alpha_ay * torch.sin(alpha_a))

		kappa_bound_d_a = (self.kappa_max * d_v**2) / torch.abs(torch.sin(alpha_a - alpha_v))
		a_max_aug = torch.minimum(self.a_max * torch.ones((self.num_batch, self.num), device=device), kappa_bound_d_a)

		d_temp_a = c2_d_a/c1_d_a
		d_a = torch.clip(d_temp_a, torch.zeros((self.num_batch, self.num), device=device), torch.tensor(self.a_max).to(device))
  
		res_ax_vec = xddot - d_a * torch.cos(alpha_a)
		res_ay_vec = yddot - d_a * torch.sin(alpha_a)
		
		res_vx_vec = xdot - d_v * torch.cos(alpha_v)
		res_vy_vec = ydot - d_v * torch.sin(alpha_v)

		res_x_obs_vec = wc_alpha - self.a_obs * d_obs * torch.cos(alpha_obs)
		res_y_obs_vec = ws_alpha - self.b_obs * d_obs * torch.sin(alpha_obs)
			
		res_vel_vec = torch.hstack([res_vx_vec,  res_vy_vec])
		res_acc_vec = torch.hstack([res_ax_vec,  res_ay_vec])
		res_obs_vec = torch.hstack([res_x_obs_vec, res_y_obs_vec])
  
		b_lane = torch.hstack(( y_ub * torch.ones((self.num_batch, self.num), device=device), -y_lb * torch.ones((self.num_batch, self.num), device=device)))
		s_lane = torch.maximum( torch.zeros((self.num_batch, 2 * self.num), device=device), -torch.mm(self.A_lane, primal_sol_y.T).T + b_lane)
		res_lane_vec = torch.mm(self.A_lane, primal_sol_y.T).T - b_lane + s_lane

		# Velocity
		vel = torch.sqrt(xdot ** 2 + ydot ** 2)
		vel_pen = torch.linalg.norm(vel - self.v_des, dim=1)

		res_norm_batch = torch.linalg.norm(res_obs_vec, dim=1) + torch.linalg.norm(res_acc_vec, dim=1) + \
						 torch.linalg.norm(res_vel_vec, dim=1) + torch.linalg.norm(res_lane_vec, dim=1) + 0.0 * vel_pen # 0.01
 
		lamda_x = lamda_x - self.rho_obs * torch.mm(self.A_obs.T, res_x_obs_vec.T).T - \
      			  self.rho_ineq * torch.mm(self.A_acc.T, res_ax_vec.T).T - \
               	  self.rho_ineq * torch.mm(self.A_vel.T, res_vx_vec.T).T
                  
		lamda_y = lamda_y - self.rho_obs * torch.mm(self.A_obs.T, res_y_obs_vec.T).T - \
      			  self.rho_ineq * torch.mm(self.A_acc.T, res_ay_vec.T).T - \
               	  self.rho_ineq * torch.mm(self.A_vel.T, res_vy_vec.T).T - \
                  self.rho_lane * torch.mm(self.A_lane.T, res_lane_vec.T).T
	
		return alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, s_lane, res_norm_batch, vel_pen

	def qp_layer_2(self, initial_state_ego, primal_sol, x_obs_traj, y_obs_traj, y_ub, y_lb, lamda_x, lamda_y): 
     
		# Boundary conditions
		b_eq_x, b_eq_y = self.compute_boundary(initial_state_ego) 
       
		# Inverse Matrices
		cost_mat_inv_x, cost_mat_inv_y = self.compute_mat_inv_2()

		# Extending Dimension
		y_ub = y_ub[:, None]
		y_lb = y_lb[:, None]
  
		alpha_obs, d_obs, alpha_a, d_a, lamda_x, lamda_y, alpha_v, d_v, s_lane, res_norm_batch, vel_pen = self.compute_alph_d(primal_sol, x_obs_traj, y_obs_traj, y_ub, y_lb, lamda_x, lamda_y)
		
		b_lane = torch.hstack([y_ub * torch.ones((self.num_batch, self.num), device=device), -y_lb * torch.ones((self.num_batch, self.num), device=device)])
		b_lane_aug = b_lane - s_lane
		b_ax_ineq = d_a * torch.cos(alpha_a)
		b_ay_ineq = d_a * torch.sin(alpha_a)
		b_vx_ineq = d_v * torch.cos(alpha_v)
		b_vy_ineq = d_v * torch.sin(alpha_v)
  
		c_x_bar = primal_sol[:,0:self.nvar]
		c_y_bar = primal_sol[:, self.nvar:]

		temp_x_obs = d_obs * torch.cos(alpha_obs) * self.a_obs
		b_obs_x = x_obs_traj + temp_x_obs
		 
		temp_y_obs = d_obs * torch.sin(alpha_obs) * self.b_obs
		b_obs_y = y_obs_traj + temp_y_obs
  
		lincost_x = -lamda_x - self.rho_projection * torch.mm(self.A_projection.T, c_x_bar.T).T - \
      				 self.rho_obs * torch.mm(self.A_obs.T, b_obs_x.T).T - \
               		 self.rho_ineq * torch.mm(self.A_acc.T, b_ax_ineq.T).T - \
                     self.rho_ineq * torch.mm(self.A_vel.T, b_vx_ineq.T).T
                     
		lincost_y = -lamda_y - self.rho_projection * torch.mm(self.A_projection.T, c_y_bar.T).T - \
      				 self.rho_obs * torch.mm(self.A_obs.T, b_obs_y.T).T - \
               		 self.rho_ineq * torch.mm(self.A_acc.T, b_ay_ineq.T).T - \
                     self.rho_ineq * torch.mm(self.A_vel.T, b_vy_ineq.T).T - \
                     self.rho_lane * torch.mm(self.A_lane.T, b_lane_aug.T).T
	
		sol_x = torch.mm(cost_mat_inv_x, torch.hstack(( -lincost_x, b_eq_x )).T).T
		sol_y = torch.mm(cost_mat_inv_y, torch.hstack(( -lincost_y, b_eq_y )).T).T

		primal_sol_x = sol_x[:,0:self.nvar]
		primal_sol_y = sol_y[:,0:self.nvar]

		primal_sol_2 = torch.hstack([primal_sol_x, primal_sol_y])
  
		return primal_sol_2, lamda_x, lamda_y, res_norm_batch, vel_pen

	# Encoder: P_phi(z | x, y)
	def _encoder(self, x, y):
		inputs = torch.cat([x, y], dim = 1)
		mean, std = self.encoder(inputs)        
		return mean, std

	# Reparametrization Trick
	def _sample_z(self, mean, std):
		eps = torch.randn_like(mean, device=device)
		return mean + std * eps

	# Decoder: P_theta(y | z, x) -> y* (init state, y)
	def _decoder(self, z, x, init_state_ego, x_obs_traj, y_obs_traj, y_ub, y_lb):
     
		inputs = torch.cat([z, x], dim = 1)
		y = self.decoder(inputs)

		# Behavioral params
		b = y[:, 0:8]
		b[:, 0:4] = torch.sigmoid(b[:, 0:4]) * 40. + 0.3

		# lamda x & y
		lamda_x = y[:, 8:19]
		lamda_y = y[:, 19:]
		
		# Call Optimization Solver First Layer
		primal_sol = self.qp_layer_1(init_state_ego, b)
  		
		res_norm_batch = 0
		# In a loop-fashion comment the lines above
		for _ in range(self.maxiter):	
			primal_sol, lamda_x, lamda_y, res_norm_batch, vel_pen = self.qp_layer_2(init_state_ego, primal_sol, x_obs_traj, y_obs_traj, y_ub, y_lb, lamda_x, lamda_y)
			res_norm_batch += res_norm_batch

		return primal_sol, self.weight_aug * res_norm_batch +  self.vel_scale * vel_pen # y_star (10**(-6))

	def cvae_loss(self, traj_sol, traj_gt, res_norm_batch, mean, std, beta = 1.0, step = 0 ):

		# Beta Annealing
		beta_d = min(step / 1000 * beta, beta)

		# Aug loss
		Aug = 0.5 * torch.mean(res_norm_batch)

		# KL Loss
		KL = -0.5 * torch.mean(torch.sum(1 + torch_log(std ** 2) - mean ** 2 - std ** 2, dim=1))
		
		# RCL Loss 
		RCL = self.rcl_loss(traj_sol, traj_gt)
					
		# ELBO Loss + Collision Cost
		loss = (RCL + beta_d * KL) + Aug

		return Aug, KL, RCL, loss

	# Forward Pass
	def forward(self, inp, traj_gt, init_state_ego, P_diag): 

		# Lane Boundaries
		y_ub = inp[:, 0]
		y_lb = inp[:, 1]

		# Batch Trajectory Prediction
		x_obs_traj, y_obs_traj = self.compute_obs_trajectories(inp)

		# Normalize input
		inp_norm = (inp - self.inp_mean) / self.inp_std
		
		# Mu & Variance
		mean, std = self._encoder(inp_norm, traj_gt)
				
		# Sample from z -> Reparameterized 
		z = self._sample_z(mean, std)
		
		# Decode y
		y_star, res_norm_batch = self._decoder(z, inp_norm, init_state_ego, x_obs_traj, y_obs_traj, y_ub, y_lb)		
		traj_sol = (P_diag @ y_star.T).T 

		return traj_sol, res_norm_batch, mean, std