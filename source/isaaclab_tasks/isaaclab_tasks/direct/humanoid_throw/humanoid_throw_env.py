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
        goal = self.catcher.data.root_pos_w + torch.tensor([0.0, 0.5, 1.0], device=self.device)
        dist = torch.norm(ball_pos - goal, dim=-1)
        reward = 1.0 - torch.tanh(dist)
        return {"thrower": reward, "catcher": reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        ball_pos = self.ball.data.root_pos_w
        fallen = ball_pos[:, 2] < 0.2
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


@torch.jit.script
def scale(x, lower, upper):
    return 0.5 * (x + 1.0) * (upper - lower) + lower


@torch.jit.script
def unscale(x, lower, upper):
    return (2.0 * x - upper - lower) / (upper - lower)
