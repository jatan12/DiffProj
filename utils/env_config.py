import  gym

env_kwargs = {
    'config': {
        "lanes_count": 4,
        "vehicles_count": 30,
        "vehicles_density": 1.0,
        "controlled_vehicles": 1,
        "initial_lane_id": None,
        "ego_spacing": 2.0,
        "duration": 50,
        "speed_limit": 15,
        # choose one  of the following
        "other_vehicles_type": "highway_env.vehicle.behavior.IDMVehicle",
        #"other_vehicles_type": "highway_env.vehicle.behavior_bgap.AggressiveCar",
        #"other_vehicles_type": "highway_env.vehicle.behavior_bgap.VeryAggressiveCar",
        "simulation_frequency": 100,  # [Hz]
        "policy_frequency": 25,  # [Hz]
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 11,
            "features": [
                "x",
                "y",
                "vx",
                "vy",
                "heading"
            ],
            "features_range": {
                "x": [-150, 150],
                "y": [-10, 10],
                "vx": [0, 30],
                "vy": [-10, 10]
            },

            "absolute": True,
            "clip": False,
            "normalize": False,
            "see_behind": True
        },
        "action": {
            "type": "PlanningAction",
            "size": 22,
        },
        "right_lane_reward": 0,
        "high_speed_reward": 0.4,
        "collision_reward": -10,
        "lane_change_reward": 0,
        "reward_speed_range": [20,33],
        "offroad_terminal": False,
        "screen_width": 1024,
        "screen_height": 800

    }
}