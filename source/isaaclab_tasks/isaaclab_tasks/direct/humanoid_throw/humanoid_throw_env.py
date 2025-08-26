# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import quat_apply

from .humanoid_throw_env_cfg import HumanoidThrowEnvCfg


class HumanoidThrowEnv(DirectMARLEnv):
    """Environment where two humanoids throw a ball to each other."""

    cfg: HumanoidThrowEnvCfg

    def __init__(self, cfg: HumanoidThrowEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.num_dofs = self.thrower.num_joints
        self.thrower_targets = torch.zeros((self.num_envs, self.num_dofs), device=self.device)
        self.catcher_targets = torch.zeros((self.num_envs, self.num_dofs), device=self.device)

        joint_limits = self.thrower.root_physx_view.get_dof_limits().to(self.device)
        self.dof_lower = joint_limits[..., 0]
        self.dof_upper = joint_limits[..., 1]

        self.ball_init_pos = torch.tensor([0.0, -0.5, 1.5], dtype=torch.float, device=self.device)
        self.catch_radius = self.cfg.catch_radius
        self.target_separation = self.cfg.target_separation
        self.termination_height = self.cfg.termination_height
        self.last_catcher = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.num_throws = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        self.up_axis = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float, device=self.device).repeat(self.num_envs, 1)

    #
    # implementation details
    #
    def _setup_scene(self):
        self.thrower = Articulation(self.cfg.thrower_robot_cfg)
        self.catcher = Articulation(self.cfg.catcher_robot_cfg)
        self.ball = RigidObject(self.cfg.ball_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        self.scene.articulations["thrower"] = self.thrower
        self.scene.articulations["catcher"] = self.catcher
        self.scene.rigid_objects["ball"] = self.ball
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = actions

    def _apply_action(self) -> None:
        self.thrower_targets = scale(self.actions["thrower"], self.dof_lower, self.dof_upper)
        self.catcher_targets = scale(self.actions["catcher"], self.dof_lower, self.dof_upper)
        self.thrower.set_joint_position_target(self.thrower_targets)
        self.catcher.set_joint_position_target(self.catcher_targets)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        ball_pos = self.ball.data.root_pos_w - self.scene.env_origins
        ball_vel = self.ball.data.root_lin_vel_w
        obs_thrower = torch.cat(
            (
                unscale(self.thrower.data.joint_pos, self.dof_lower, self.dof_upper),
                self.cfg.vel_obs_scale * self.thrower.data.joint_vel,
                ball_pos,
                ball_vel,
            ),
            dim=-1,
        )
        obs_catcher = torch.cat(
            (
                unscale(self.catcher.data.joint_pos, self.dof_lower, self.dof_upper),
                self.cfg.vel_obs_scale * self.catcher.data.joint_vel,
                ball_pos,
                ball_vel,
            ),
            dim=-1,
        )
        return {"thrower": obs_thrower, "catcher": obs_catcher}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        ball_pos = self.ball.data.root_pos_w
        thrower_root = self.thrower.data.root_pos_w
        catcher_root = self.catcher.data.root_pos_w
        thrower_goal = thrower_root + torch.tensor([0.0, 0.5, 1.0], device=self.device)
        catcher_goal = catcher_root + torch.tensor([0.0, 0.5, 1.0], device=self.device)
        dist_thrower = torch.norm(ball_pos - thrower_goal, dim=-1)
        dist_catcher = torch.norm(ball_pos - catcher_goal, dim=-1)

        last_holder = self.last_catcher.clone()
        catch_thrower = dist_thrower < self.catch_radius
        catch_catcher = dist_catcher < self.catch_radius
        self.last_catcher = torch.where(catch_thrower, torch.zeros_like(self.last_catcher), self.last_catcher)
        self.last_catcher = torch.where(catch_catcher, torch.ones_like(self.last_catcher), self.last_catcher)
        new_catch = ((last_holder == 0) & catch_catcher) | ((last_holder == 1) & catch_thrower)
        self.num_throws += new_catch.float()
        throws_reward = new_catch.float()

        goal_dist = torch.where(last_holder == 0, dist_catcher, dist_thrower)
        approach_reward = 1.0 - torch.tanh(goal_dist)

        separation = torch.norm(thrower_root - catcher_root, dim=-1)
        distance_reward = 1.0 - torch.tanh(torch.abs(separation - self.target_separation))

        thrower_up = quat_apply(self.thrower.data.root_quat_w, self.up_axis)[..., 2]
        catcher_up = quat_apply(self.catcher.data.root_quat_w, self.up_axis)[..., 2]
        upright = 0.5 * (thrower_up + catcher_up).clamp(min=0.0)
        upright_reward = self.cfg.upright_reward_scale * upright

        ball_fallen = ball_pos[:, 2] < self.cfg.ball_ground_height
        thrower_fallen = thrower_root[:, 2] < self.termination_height
        catcher_fallen = catcher_root[:, 2] < self.termination_height
        alive = ~(ball_fallen | thrower_fallen | catcher_fallen)
        alive_reward = self.cfg.alive_reward_scale * alive.float()

        reward = (
            approach_reward
            + throws_reward
            + self.cfg.dist_reward_scale * distance_reward
            + upright_reward
            + alive_reward
        )
        return {"thrower": reward, "catcher": reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ball_pos = self.ball.data.root_pos_w
        thrower_pos = self.thrower.data.root_pos_w
        catcher_pos = self.catcher.data.root_pos_w
        ball_fallen = ball_pos[:, 2] < self.cfg.ball_ground_height
        thrower_fallen = thrower_pos[:, 2] < self.termination_height
        catcher_fallen = catcher_pos[:, 2] < self.termination_height
        fallen = ball_fallen | thrower_fallen | catcher_fallen
        terminated = {agent: fallen for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.ball._ALL_INDICES
        super()._reset_idx(env_ids)
        self.thrower.reset(env_ids)
        self.catcher.reset(env_ids)
        ball_state = self.ball.data.default_root_state[env_ids]
        ball_state[:, :3] = self.ball_init_pos + self.scene.env_origins[env_ids]
        ball_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        ball_state[:, 7:] = 0
        self.ball.write_root_pose_to_sim(ball_state[:, :7], env_ids)
        self.ball.write_root_velocity_to_sim(ball_state[:, 7:], env_ids)
        self.last_catcher[env_ids] = 0
        self.num_throws[env_ids] = 0


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
