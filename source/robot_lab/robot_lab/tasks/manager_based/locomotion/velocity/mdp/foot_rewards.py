"""Foot contact and swing measurements against the static locomotion terrain."""

from functools import cached_property
from itertools import product

import numpy as np
import torch
import warp as wp
from pxr import Gf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull, QhullError

from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.utils.math import matrix_from_quat, quat_apply
from isaaclab.utils.warp import convert_to_warp_mesh, raycast_mesh


@wp.kernel
def _contact_motion(
    state: wp.array2d(dtype=wp.float32),
    forces: wp.array(dtype=wp.float32),
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    counts: wp.array2d(dtype=wp.int32),
    offsets: wp.array2d(dtype=wp.int32),
    foot: int,
    load: wp.array2d(dtype=wp.float32),
    speed: wp.array2d(dtype=wp.float32),
):
    i = wp.tid()
    origin = wp.vec3(state[i, 0], state[i, 1], state[i, 2])
    linear = wp.vec3(state[i, 7], state[i, 8], state[i, 9])
    angular = wp.vec3(state[i, 10], state[i, 11], state[i, 12])
    total = float(0.0)
    weighted_speed = float(0.0)
    for ground in range(counts.shape[1]):
        start = offsets[i, ground]
        for j in range(start, start + counts[i, ground]):
            fz = wp.max(forces[j] * normals[j][2], 0.0)
            velocity = linear + wp.cross(angular, points[j] - origin)
            total += fz
            weighted_speed += fz * wp.length(wp.vec2(velocity[0], velocity[1]))
    load[i, foot] = total
    speed[i, foot] = weighted_speed / total if total > 0.0 else 0.0


def _terrain_colliders(env):
    root = env.sim.stage.GetPrimAtPath(env.cfg.scene.terrain.prim_path)
    colliders = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies()) if p.HasAPI(UsdPhysics.CollisionAPI)]
    if not colliders:
        raise ValueError(f"Missing terrain collider under {root.GetPath()}")
    return colliders


def _terrain_height_geometry(env, height_scanner_cfg):
    cache = UsdGeom.XformCache()
    grounds = _terrain_colliders(env)
    scanner_name = height_scanner_cfg.name if height_scanner_cfg is not None else "height_scanner_base"
    scanner = env.scene.sensors.get(scanner_name)
    cached_mesh = getattr(scanner, "meshes", {}).get(env.cfg.scene.terrain.prim_path)
    plane_height = -float("inf")
    meshes = []
    for ground in grounds:
        transform = cache.GetLocalToWorldTransform(ground)
        if ground.IsA(UsdGeom.Plane):
            if (ground.GetAttribute("axis").Get() != "Z" or
                    not np.allclose(transform.TransformDir(Gf.Vec3d(0, 0, 1)).GetNormalized(), (0, 0, 1))):
                raise ValueError(f"Expected a horizontal terrain plane at {ground.GetPath()}")
            plane_height = max(plane_height, float(transform.ExtractTranslation()[2]))
        elif len(grounds) == 1 and cached_mesh is not None:
            meshes.append(cached_mesh)
        else:
            mesh = UsdGeom.Mesh(ground)
            if not mesh or not np.all(np.asarray(mesh.GetFaceVertexCountsAttr().Get()) == 3):
                raise ValueError(f"Expected a triangle terrain mesh at {ground.GetPath()}")
            matrix = np.asarray(transform)
            points = np.asarray(mesh.GetPointsAttr().Get()) @ matrix[:3, :3] + matrix[3, :3]
            meshes.append(convert_to_warp_mesh(points, np.asarray(mesh.GetFaceVertexIndicesAttr().Get()),
                                               device=env.device))
    return plane_height, meshes


def terrain_height_below(env, points_w, height_scanner_cfg=None):
    """Cast down at the current points, reusing the static terrain mesh across reward terms.

    A height scanner can supply its cached mesh. A ray that misses returns negative infinity.
    """
    if not hasattr(env, "_foot_terrain_geometry"):
        env._foot_terrain_geometry = _terrain_height_geometry(env, height_scanner_cfg)
    plane_height, meshes = env._foot_terrain_geometry
    floor = torch.full_like(points_w[..., 2], plane_height)
    if meshes:
        starts = points_w.clone()
        starts[..., 2] += 1.0  # Start above ground even if a sole is penetrating it.
        down = torch.zeros_like(starts)
        down[..., 2] = -1
        for mesh in meshes:
            height = raycast_mesh(starts, down, mesh)[0][..., 2]
            floor = torch.where(torch.isfinite(height), torch.maximum(floor, height), floor)
    return floor


class _FootGround:
    """One terrain-filtered contact view per foot, shared by both reward terms."""

    def __init__(self, env, robot, body_ids):
        self.robot = robot
        self.body_ids = list(body_ids)
        ground_paths = [str(p.GetPath()) for p in _terrain_colliders(env)]
        links = robot.root_physx_view.link_paths[0]
        self.link_paths = [links[body] for body in body_ids]
        first_env = env.scene.env_prim_paths[0]
        env_pattern = env.scene.env_regex_ns.replace(".*", "*")
        self.views = [
            env.sim.physics_sim_view.create_rigid_contact_view(
                path.replace(first_env + "/", env_pattern + "/", 1), ground_paths,
                max_contact_data_count=128 * env.num_envs,
            ) for path in self.link_paths
        ]
        if any(view.sensor_count != env.num_envs or view.filter_count != len(ground_paths) for view in self.views):
            raise ValueError(f"Could not bind feet {self.link_paths} to terrain {ground_paths} in every environment")
        self.load = torch.zeros((env.num_envs, len(body_ids)), device=env.device)
        self.speed = torch.zeros_like(self.load)
        self.step = -1

    def update(self, env):
        if self.step == env.common_step_counter:
            return
        state = self.robot.data.body_link_state_w
        for foot, (body, view) in enumerate(zip(self.body_ids, self.views)):
            forces, points, normals, _, counts, offsets = view.get_contact_data(env.physics_dt)
            wp.launch(
                _contact_motion, dim=env.num_envs,
                inputs=[wp.from_torch(state[:, body]), wp.from_torch(forces.view(-1)),
                        wp.from_torch(points, dtype=wp.vec3), wp.from_torch(normals, dtype=wp.vec3),
                        wp.from_torch(counts), wp.from_torch(offsets), foot],
                outputs=[wp.from_torch(self.load), wp.from_torch(self.speed)], device=env.device,
                stream=wp.stream_from_torch(state.device) if state.is_cuda else None,
            )
        self.step = env.common_step_counter

    def slip(self, env, support_force):
        self.update(env)
        return (self.speed * (self.load > support_force)).sum(dim=1)


def _ground_for(env, asset_cfg, sensor_cfg=None):
    robot = env.scene[asset_cfg.name]
    body_ids = asset_cfg.body_ids
    # Older slide configs may specify the feet only on the contact sensor.
    if isinstance(body_ids, slice) and body_ids == slice(None) and sensor_cfg is not None:
        sensor_names = env.scene[sensor_cfg.name].body_names
        ids = sensor_cfg.body_ids
        names = sensor_names[ids] if isinstance(ids, slice) else [sensor_names[i] for i in ids]
        body_ids = robot.find_bodies(names, preserve_order=True)[0]
    if isinstance(body_ids, slice):
        body_ids = range(robot.num_bodies)[body_ids]
    key = (asset_cfg.name, tuple(body_ids))
    if not hasattr(env, "_foot_ground_contacts"):
        env._foot_ground_contacts = {}
    if key not in env._foot_ground_contacts:
        env._foot_ground_contacts[key] = _FootGround(env, robot, key[1])
    return env._foot_ground_contacts[key]


class GroundContactSlip(ManagerTermBase):
    """Sum of per-foot load-weighted horizontal contact point speeds in m/s."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contacts = _ground_for(env, cfg.params["asset_cfg"], cfg.params["sensor_cfg"])

    def __call__(self, env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg, support_force: float = 5.0):
        return self.contacts.slip(env, support_force)


class _Sole:
    """Support point in the world-down direction for the link's collision geometry."""

    def __init__(self, link, cache, device):
        colliders = []
        tree = iter(Usd.PrimRange(link, Usd.TraverseInstanceProxies()))
        for prim in tree:
            if prim != link and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                tree.PruneChildren()
            elif prim.HasAPI(UsdPhysics.CollisionAPI):
                colliders.append(prim)
        shapes = {p.GetPath(): p for collider in colliders
                  for p in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()) if p.IsA(UsdGeom.Gprim)}
        inverse_link = cache.GetLocalToWorldTransform(link).GetInverse()
        vertices = []
        self.round_shapes = []
        for shape in shapes.values():
            transform = np.asarray(cache.GetLocalToWorldTransform(shape) * inverse_link)
            kind = shape.GetTypeName()
            if kind == "Mesh":
                points = np.asarray(UsdGeom.Mesh(shape).GetPointsAttr().Get())
            elif kind == "Cube":
                points = np.array(list(product((-0.5, 0.5), repeat=3))) * shape.GetAttribute("size").Get()
            elif kind in ("Sphere", "Capsule", "Cylinder"):
                radius = float(shape.GetAttribute("radius").Get())
                axis = 2 if kind == "Sphere" else "XYZ".index(shape.GetAttribute("axis").Get())
                half = 0.0 if kind == "Sphere" else float(shape.GetAttribute("height").Get()) / 2
                matrix = torch.tensor(transform, dtype=torch.float32, device=device)
                self.round_shapes.append((kind, radius, axis, half, matrix))
                continue
            else:
                raise ValueError(f"Unsupported sole collider {kind} at {shape.GetPath()}")
            vertices.append(points @ transform[:3, :3] + transform[3, :3])
        if not vertices and not self.round_shapes:
            raise ValueError(f"No sole collision geometry at {link.GetPath()}")
        self.vertices = None
        if vertices:
            points = np.unique(np.concatenate(vertices), axis=0)
            try:
                points = points[ConvexHull(points).vertices]
            except QhullError:  # A planar mesh has no 3-D hull; its vertices still define its support.
                pass
            self.vertices = torch.as_tensor(points, dtype=torch.float32, device=device)

    def lowest(self, pose, rotation_z):
        candidates = []
        if self.vertices is not None:
            lowest = (rotation_z @ self.vertices.T).argmin(dim=1)
            candidates.append(self.vertices[lowest])
        for kind, radius, axis, half, matrix in self.round_shapes:
            direction = -rotation_z @ matrix[:3, :3].T
            end = torch.sign(direction[:, axis]) * half
            if kind == "Cylinder":
                direction[:, axis] = 0
            support = radius * torch.nn.functional.normalize(direction, dim=1)
            support[:, axis] += end
            candidates.append(support @ matrix[:3, :3] + matrix[3, :3])
        point = candidates[0]
        if len(candidates) > 1:
            points = torch.stack(candidates, dim=1)
            lowest = (points * rotation_z.unsqueeze(1)).sum(dim=2).argmin(dim=1)
            point = points.gather(1, lowest[:, None, None].expand(-1, 1, 3)).squeeze(1)
        return pose[:, :3] + quat_apply(pose[:, 3:], point)


def swing_clearance_score(gap, load, episode_steps, step_dt, command, gait_params, target_height, std, support_force):
    airborne = load <= support_force
    if gait_params is None:
        swing = airborne
        target = target_height
    else:
        frequency, offset, duty = gait_params
        phase = (episode_steps * step_dt * frequency).remainder(1.0)
        phases = torch.stack((phase, (phase + offset).remainder(1.0)), dim=1)
        swing = phases >= duty
        progress = ((phases - duty) / (1.0 - duty)).clamp(0.0, 1.0)
        target = target_height * torch.sin(torch.pi * progress)
    eligible = swing & airborne & (gap > 0) & torch.isfinite(gap)
    score = torch.exp(-((gap - target) / std).square())
    # A scheduled swing foot that stays planted earns zero and still counts in the denominator.
    score = torch.where(eligible, score, 0.0).sum(dim=1) / swing.sum(dim=1).clamp_min(1)
    return score * (command.norm(dim=1) > 0.1)


class SwingSoleClearance(ManagerTermBase):
    """Track sole-to-local-ground clearance during the commanded gait's swing."""

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.contacts = _ground_for(env, cfg.params["asset_cfg"])
        cache = UsdGeom.XformCache()
        self.soles = [_Sole(env.sim.stage.GetPrimAtPath(path), cache, env.device) for path in self.contacts.link_paths]
        self.gait_params = None
        self.gait_order = slice(None)
        gait = getattr(env.cfg.rewards, "bipedal_gait_reward", None)
        if gait is not None:
            selector = gait.params["asset_cfg"]
            gait_ids, _ = self.contacts.robot.find_bodies(selector.body_names, preserve_order=selector.preserve_order)
            if len(gait_ids) == 2 and set(gait_ids) == set(self.contacts.body_ids):
                self.gait_params = gait.params["gait_params"]
                if gait_ids != self.contacts.body_ids:
                    self.gait_order = [self.contacts.body_ids.index(i) for i in gait_ids]
        self.contacts.clearance_term = self

    @cached_property
    def legacy_sole_offset(self):
        """Convert a historical link-height target to sole height in gait order."""
        landing = getattr(self._env.cfg.rewards, "feet_landing_vel", None)
        if landing is not None and "foot_radius" in landing.params:
            selector = landing.params["asset_cfg"]
            current = self.cfg.params["asset_cfg"]
            if (selector.name, selector.body_names) == (current.name, current.body_names):
                return -landing.params["foot_radius"]
        # Configurations without an explicit nominal radius use their neutral link geometry.
        pose = self.contacts.load.new_tensor([[0., 0., 0., 1., 0., 0., 0.]])
        up = pose.new_tensor([[0., 0., 1.]])
        return torch.stack([sole.lowest(pose, up)[0, 2] for sole in self.soles])[self.gait_order]

    def __call__(self, env, command_name: str, asset_cfg: SceneEntityCfg, target_height: float, std: float,
                 support_force: float = 5.0):
        self.contacts.update(env)
        poses = self.contacts.robot.data.body_link_pose_w[:, self.contacts.body_ids]
        rotation_z = matrix_from_quat(poses[..., 3:])[..., 2, :]
        sole = torch.stack([s.lowest(poses[:, i], rotation_z[:, i]) for i, s in enumerate(self.soles)], dim=1)
        self.gap = sole[..., 2] - terrain_height_below(env, sole)
        return swing_clearance_score(
            self.gap[:, self.gait_order], self.contacts.load[:, self.gait_order], env.episode_length_buf, env.step_dt,
            env.command_manager.get_command(command_name), self.gait_params, target_height, std, support_force,
        )


def feet_slide(env, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
               support_force: float = 5.0):
    """Functional entry point retained for existing callers and saved manager configurations."""
    return _ground_for(env, asset_cfg, sensor_cfg).slip(env, support_force)


def feet_clearance_reward(env, command_name: str, asset_cfg: SceneEntityCfg, target_height: float, std: float,
                          tanh_mult: float | None = None, support_force: float = 5.0):
    """The retired tanh_mult argument identifies saved configurations with link-height targets."""
    contacts = _ground_for(env, asset_cfg)
    if not hasattr(contacts, "clearance_term"):
        SwingSoleClearance(RewardTermCfg(func=SwingSoleClearance, weight=1.0, params={"asset_cfg": asset_cfg}), env)
    if tanh_mult is not None:
        target_height = target_height + contacts.clearance_term.legacy_sole_offset
    return contacts.clearance_term(env, command_name, asset_cfg, target_height, std, support_force)
