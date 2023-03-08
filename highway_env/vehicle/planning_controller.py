from typing import List, Union, Optional
import numpy as np
from highway_env.road.road import Road, LaneIndex, Route
from highway_env.utils import Vector
from highway_env.vehicle.controller import MDPVehicle
from utils import bernstein_coeff_order10_arbitinterval


class PlanningVehicle(MDPVehicle):
    def __init__(self,
                 road: Road,
                 position: List[float],
                 heading: float = 0,
                 speed: float = 0,
                 target_lane_index: Optional[LaneIndex] = None,
                 target_speed: Optional[float] = None,
                 target_speeds: Optional[Vector] = None,
                 route: Optional[Route] = None) -> None:

        super().__init__(road, position, heading, speed, target_lane_index, target_speed, target_speeds, route)


        t_fin = 15.0
        self.num_up = 1500
        self.Ts = t_fin / self.num_up
        tot_time_up = np.linspace(0, t_fin, self.num_up).reshape(self.num_up, 1)
        self.P_up, self.Pdot_up, self.Pddot_up = bernstein_coeff_order10_arbitinterval.bernstein_coeff_order10_new(10,
                                                                                                                   tot_time_up[
                                                                                                                       0],
                                                                                                                   tot_time_up[
                                                                                                                       -1],
                                                                                                                   tot_time_up)
        self.cnt = 0
        self.a_controls = np.zeros(shape=self.num_up)
        self.steer_controls = np.zeros(shape=self.num_up)


    def act(self, action: Union[dict, str] = None) -> None:
        if action is not None:
            c_x = action[0:11]
            c_y = action[11:]

            a_best, steer_best = self.compute_controls(c_x, c_y)
            a_best_np = np.asarray(a_best)
            steer_best_np = np.clip(np.asarray(steer_best),-0.6,0.6)

            self.a_controls = a_best_np
            self.steer_controls = steer_best_np
            self.cnt = 0

        else:
            action = {"steering": self.steer_controls[self.cnt],
                    "acceleration": self.a_controls[self.cnt]}
            self.action = action
            self.cnt += 1


    def compute_controls(self, c_x_best, c_y_best):

        xdot_best = np.dot(self.Pdot_up, c_x_best)
        ydot_best = np.dot(self.Pdot_up, c_y_best)

        xddot_best = np.dot(self.Pddot_up, c_x_best)
        yddot_best = np.dot(self.Pddot_up, c_y_best)

        curvature_best = (yddot_best * xdot_best - ydot_best * xddot_best) / (
                    (xdot_best ** 2 + ydot_best ** 2) ** (1.5))
        steer_best = np.arctan(curvature_best * self.LENGTH / 2)

        v_best = np.sqrt(xdot_best ** 2 + ydot_best ** 2)
        a_best = np.diff(v_best, axis=0) / self.Ts

        return a_best, steer_best









