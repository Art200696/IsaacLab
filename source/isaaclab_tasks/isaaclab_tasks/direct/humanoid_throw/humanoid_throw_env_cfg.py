# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import HUMANOID_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class HumanoidThrowEnvCfg(DirectMARLEnvCfg):
    """Configuration for humanoid ball throw environment."""

    # env
    decimation = 2
    episode_length_s = 10.0
    possible_agents = ["thrower", "catcher"]
    action_spaces = {"thrower": 21, "catcher": 21}
    observation_spaces = {"thrower": 48, "catcher": 48}
    state_space = 0
    vel_obs_scale = 0.1
    catch_radius = 0.3
    target_separation = 2.0
    dist_reward_scale = 0.1
    upright_reward_scale = 0.1
    alive_reward_scale = 0.1
    termination_height = 0.5
    ball_ground_height = 0.1

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robots
    thrower_robot_cfg: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Thrower").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 1.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )
    catcher_robot_cfg: ArticulationCfg = HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Catcher").replace(
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -2.0, 1.0),
            rot=(0.0, 0.0, 1.0, 0.0),
            joint_pos={".*": 0.0},
        )
    )

    # ball object
    ball_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/ball",
        spawn=sim_utils.SphereCfg(
            radius=0.1,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.1, 0.1)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=0.7, dynamic_friction=0.5),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=500.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=8.0, replicate_physics=True)
