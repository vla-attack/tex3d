import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import sys
import math
import time
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import tqdm
from PIL import Image
from scipy.spatial.transform import Rotation as R
import torch
import torch.nn as nn
import torch.nn.functional as F
import trimesh
import nvdiffrast.torch as dr
import draccus
from omegaconf import OmegaConf

PATH_TO_LIBERO_ROOT = "/path/to/LIBERO"
if PATH_TO_LIBERO_ROOT not in sys.path:
    sys.path.append(PATH_TO_LIBERO_ROOT)

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path

try:
    from openpi.policies import policy_config
    from openpi.training import config as training_config
    import openpi.models.model as _model
except ImportError:
    raise ImportError("请确保已在 openpi 环境下运行此脚本")

ASSET_ROOT_SCANNED = "/path/to/LIBERO/libero/libero/assets/stable_scanned_objects"
ASSET_ROOT_HOPE    = "/path/to/LIBERO/libero/libero/assets/stable_hope_objects"


def _scanned(name, mesh_file=None, tex_file="texture.png"):
    base = f"{ASSET_ROOT_SCANNED}/{name}"
    return {
        "xml":     f"{base}/{name}.xml",
        "mesh":    f"{base}/{mesh_file or name + '.obj'}",
        "texture": f"{base}/{tex_file}",
    }


def _hope(name, mesh_file=None, tex_file="texture_map.png"):
    base = f"{ASSET_ROOT_HOPE}/{name}"
    return {
        "xml":     f"{base}/{name}.xml",
        "mesh":    f"{base}/{mesh_file or 'textured.obj'}",
        "texture": f"{base}/{tex_file}",
    }


OBJECTS = {
    "akita_black_bowl": {
        **_scanned("akita_black_bowl"),
        "search":     [["akita_black_bowl"], ["bowl"]],
        "task_suite": "libero_spatial",
        "task_id":    0,
    },
    "alphabet_soup": {
        **_hope("alphabet_soup", mesh_file="alphabet_soup_col.obj"),
        "search":     [["alphabet_soup"], ["soup"]],
        "task_suite": "libero_object",
        "task_id":    0,
    },
    "cream_cheese": {
        **_hope("cream_cheese", mesh_file="hope_cream_cheese_coll.obj"),
        "search":     [["cream_cheese"], ["cheese"]],
        "task_suite": "libero_object",
        "task_id":    1,
    },
    "salad_dressing": {
        **_hope("salad_dressing", mesh_file="salad_dressing_col.obj"),
        "search":     [["salad_dressing"], ["dressing"]],
        "task_suite": "libero_object",
        "task_id":    2,
    },
    "bbq_sauce": {
        **_hope("bbq_sauce", mesh_file="hope_bbq_coll.obj"),
        "search":     [["bbq_sauce"], ["bbq"], ["sauce"]],
        "task_suite": "libero_object",
        "task_id":    3,
    },
    "ketchup": {
        **_hope("ketchup", mesh_file="ketchup_col.obj"),
        "search":     [["ketchup"]],
        "task_suite": "libero_object",
        "task_id":    4,
    },
    "tomato_sauce": {
        **_hope("tomato_sauce"),
        "search":     [["tomato_sauce"], ["tomato"]],
        "task_suite": "libero_object",
        "task_id":    5,
    },
    "butter": {
        **_hope("butter", mesh_file="hope_butter_coll.obj"),
        "search":     [["butter"]],
        "task_suite": "libero_object",
        "task_id":    6,
    },
    "milk": {
        **_hope("milk", mesh_file="milk_col.obj"),
        "search":     [["milk"]],
        "task_suite": "libero_object",
        "task_id":    7,
    },
    "chocolate_pudding": {
        **_hope("chocolate_pudding", mesh_file="chocolate_pudding_col.obj"),
        "search":     [["chocolate_pudding"], ["chocolate"], ["pudding"]],
        "task_suite": "libero_object",
        "task_id":    8,
    },
    "orange_juice": {
        **_hope("orange_juice", mesh_file="orange_juice_col.obj"),
        "search":     [["orange_juice"], ["orange"], ["juice"]],
        "task_suite": "libero_object",
        "task_id":    9,
    },
}


def parse_mesh_scale(xml_path: str) -> List[float]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for mesh_elem in root.findall(".//mesh"):
        scale_str = mesh_elem.get("scale")
        if scale_str:
            vals = [float(v) for v in scale_str.strip().split()]
            return vals if len(vals) == 3 else [vals[0]] * 3
    return [1.0, 1.0, 1.0]


def get_target_model_matrix(env, search_keywords_list: List[List[str]]):
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    target_body_id = -1
    found_name = None

    for keywords in search_keywords_list:
        for i in range(sim.model.nbody):
            try:
                name = sim.model.body_id2name(i)
            except Exception:
                name = None
            if not name or "vis" in name or "site" in name:
                continue
            if all(k in name for k in keywords):
                target_body_id = i
                found_name = name
                break
        if target_body_id != -1:
            break

    if target_body_id == -1:
        return None, -1, None

    pos  = sim.data.body_xpos[target_body_id]
    quat = sim.data.body_xquat[target_body_id]
    rot  = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat  = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    mat[:3,  3] = pos
    return torch.from_numpy(mat).cuda(), target_body_id, found_name


def build_mvp(env, model_matrix: torch.Tensor, resolution=(256, 256)) -> torch.Tensor:
    sim = env.sim if not hasattr(env, "unwrapped") else env.unwrapped.sim
    W, H = resolution

    cam_id = 0
    try:
        cam_id = sim.model.camera_name2id("agentview")
    except Exception:
        pass

    cam_pos  = sim.data.cam_xpos[cam_id]
    cam_xmat = sim.data.cam_xmat[cam_id].reshape(3, 3)

    view_rot = cam_xmat.T
    view_mtx = np.eye(4, dtype=np.float32)
    view_mtx[:3, :3] = view_rot
    view_mtx[:3,  3] = -(view_rot @ cam_pos)

    fovy_deg = float(sim.model.cam_fovy[cam_id])
    aspect   = float(W) / float(H)
    near, far = 0.01, 10.0
    f = 1.0 / np.tan(np.deg2rad(fovy_deg) / 2.0)

    proj_mtx = np.zeros((4, 4), dtype=np.float32)
    proj_mtx[0, 0] = -f / aspect
    proj_mtx[1, 1] = -f
    proj_mtx[2, 2] = (far + near) / (near - far)
    proj_mtx[2, 3] = (2 * far * near) / (near - far)
    proj_mtx[3, 2] = -1.0

    return (
        torch.from_numpy(proj_mtx).cuda()
        @ torch.from_numpy(view_mtx).cuda()
        @ model_matrix
    )


def get_libero_dummy_action():
    return [0.0] * 6 + [-1.0]


def _get_libero_env(task, resolution, seed=7):
    task_description = task.language
    task_bddl_file = (
        Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths":  resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task_description


def get_libero_image(obs, resolution=256):
    return np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])


def _quat2axisangle(quat):
    quat = np.clip(quat, -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if np.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * np.arccos(quat[3])) / den


def build_raw_inputs(obs_env, task_description, adv_image_np=None, resolution=256):
    img = get_libero_image(obs_env, resolution) if adv_image_np is None else adv_image_np
    wrist = np.ascontiguousarray(obs_env["robot0_eye_in_hand_image"][::-1, ::-1])
    state = np.concatenate((
        obs_env["robot0_eef_pos"],
        _quat2axisangle(obs_env["robot0_eef_quat"]),
        obs_env["robot0_gripper_qpos"],
    ))
    return {
        "observation/image":       img,
        "observation/wrist_image": wrist,
        "observation/state":       state,
        "prompt":                  str(task_description),
    }


class DifferentiableRenderer(nn.Module):
    def __init__(self, mesh_path, orig_texture_path=None, device="cuda",
                 scale_xyz=None, pos_offset=None, epsilon=128.0 / 255.0):
        super().__init__()
        self.device = device
        self.epsilon = epsilon
        self.texture_res = 256
        self.pos_offset = torch.tensor(
            pos_offset or [0.0, 0.0, 0.005], dtype=torch.float32, device=device
        )

        if scale_xyz is None:
            scale_xyz = [1.0, 1.0, 1.0]
        scale_arr = np.array(scale_xyz, dtype=np.float64)

        try:
            mesh = trimesh.load(mesh_path, force="mesh")
        except Exception:
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.001])

        vertices = mesh.vertices * scale_arr[None, :]
        self.num_vertices = len(vertices)
        self.register_buffer("pos",   torch.from_numpy(vertices.astype(np.float32)).to(device))
        self.register_buffer("faces", torch.from_numpy(mesh.faces.astype(np.int32)).to(device))

        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            uv = mesh.visual.uv.astype(np.float32)
            self.register_buffer("uv", torch.from_numpy(uv).to(device))
            if (hasattr(mesh.visual, "face_uv") and mesh.visual.face_uv is not None
                    and len(mesh.visual.face_uv) == len(mesh.faces)):
                self.register_buffer("uv_idx", torch.from_numpy(mesh.visual.face_uv.astype(np.int32)).to(device))
            else:
                self.register_buffer("uv_idx", self.faces)
        else:
            v  = torch.from_numpy(vertices).to(device)
            uv = v[:, :2]
            if uv.numel() > 0:
                mn, mx = uv.min(0, keepdim=True)[0], uv.max(0, keepdim=True)[0]
                uv = (uv - mn) / (mx - mn + 1e-8)
            else:
                uv = torch.zeros((len(vertices), 2), device=device)
            self.register_buffer("uv",     uv.float())
            self.register_buffer("uv_idx", self.faces)

        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            mesh.fix_normals()
        vn = np.array(getattr(mesh, "vertex_normals", np.zeros_like(vertices)))
        if vn.sum() == 0:
            vn[:, 2] = 1.0
        vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
        self.register_buffer("vn", torch.from_numpy(vn.astype(np.float32)).to(device))

        self.glctx = dr.RasterizeCudaContext()
        if orig_texture_path and os.path.exists(orig_texture_path):
            img = (Image.open(orig_texture_path).convert("RGB")
                        .resize((self.texture_res, self.texture_res))
                        .transpose(Image.FLIP_TOP_BOTTOM))
            tex = torch.from_numpy(np.array(img)).float() / 255.0
            self.register_buffer("orig_texture", tex.unsqueeze(0).to(device).contiguous())
        else:
            fb = torch.tensor([0.45, 0.45, 0.45], device=device).view(1, 1, 1, 3).expand(
                1, self.texture_res, self.texture_res, 3).contiguous()
            self.register_buffer("orig_texture", fb)

        self.adv_vc_noise = nn.Parameter(
            torch.zeros((self.num_vertices, 3), dtype=torch.float32, device=device)
        )
        self.light_dir = F.normalize(torch.tensor([0.5, 0.5, 1.0], device=device), dim=0)

    def get_texture_param(self):
        return self.adv_vc_noise

    def render(self, mvp, resolution=(256, 256), return_clean=False):
        pos      = self.pos + self.pos_offset
        pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())

        rast, _         = dr.rasterize(self.glctx, pos_clip.unsqueeze(0), self.faces, resolution=resolution)
        tex_uv, _       = dr.interpolate(self.uv.unsqueeze(0), rast, self.uv_idx)
        clean_color     = dr.texture(self.orig_texture.contiguous(), tex_uv.contiguous(), filter_mode="linear")

        noise           = torch.tanh(self.adv_vc_noise) * self.epsilon
        noise_interp, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast, self.faces)
        adv_color       = torch.clamp(clean_color + noise_interp, 0.0, 1.0)

        vn_interp, _ = dr.interpolate(self.vn.unsqueeze(0), rast, self.faces)
        vn_interp    = F.normalize(vn_interp, p=2, dim=-1)
        diffuse      = torch.clamp((vn_interp * self.light_dir.view(1, 1, 1, 3)).sum(-1, keepdim=True), 0.0, 1.0)
        lighting     = 0.6 + 0.4 * diffuse

        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            return adv_color * lighting, clean_color * lighting, mask
        return adv_color * lighting, mask

    def get_baked_adv_texture(self):
        with torch.no_grad():
            uv_clip = self.uv * 2.0 - 1.0
            uv4     = torch.cat([uv_clip, torch.zeros_like(uv_clip[..., :1]),
                                  torch.ones_like(uv_clip[..., :1])], dim=-1)
            rast, _ = dr.rasterize(self.glctx, uv4.unsqueeze(0), self.uv_idx,
                                    resolution=(self.texture_res, self.texture_res))
            noise    = torch.tanh(self.adv_vc_noise) * self.epsilon
            baked, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast, self.faces)
            mask     = (rast[..., 3] > 0).float().unsqueeze(-1)
            result   = torch.clamp(self.orig_texture + baked * mask, 0.0, 1.0)
            return torch.flip(result, dims=[1])


def hide_object_geoms(sim, search_keywords_list: List[List[str]]):
    hidden = []
    for i in range(sim.model.ngeom):
        try:
            name = sim.model.geom_id2name(i)
        except Exception:
            name = None
        if not name:
            continue
        matched = any(all(k in name for k in kws) for kws in search_keywords_list)
        if matched:
            orig = sim.model.geom_rgba[i].copy()
            sim.model.geom_rgba[i, 3] = 0.0
            hidden.append((i, orig))
    return hidden


def restore_object_geoms(sim, hidden):
    for i, orig in hidden:
        sim.model.geom_rgba[i] = orig


def find_object_tex_id(sim, search_keywords_list: List[List[str]]):
    for i in range(sim.model.ntex):
        name = None
        try:
            name = sim.model.tex_id2name(i) if hasattr(sim.model, "tex_id2name") else None
        except Exception:
            pass
        if name and any(all(k in name for k in kws) for kws in search_keywords_list):
            return i
    for i in range(sim.model.nmat):
        mat_name = None
        try:
            mat_name = sim.model.mat_id2name(i) if hasattr(sim.model, "mat_id2name") else None
        except Exception:
            pass
        if mat_name and any(all(k in mat_name for k in kws) for kws in search_keywords_list):
            try:
                tex_id = int(sim.model.mat_texid[i, 0])
                if tex_id >= 0:
                    return tex_id
            except Exception:
                pass
    return -1


def render_hires(renderer, bg_tensor, mvp, hires=1024, render_res=256):
    if mvp is None:
        return bg_tensor
    bg_hires = F.interpolate(bg_tensor, size=(hires, hires), mode="bilinear", align_corners=False)
    adv_rgba, mask = renderer.render(mvp, resolution=(hires, hires))
    adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
    mask_t   = mask.permute(0, 3, 1, 2)
    return torch.clamp(adv_rgb * mask_t + bg_hires * (1 - mask_t), 0.0, 1.0)


def save_hires(tensor, path, hires=1024):
    if tensor.shape[-1] != hires or tensor.shape[-2] != hires:
        tensor = F.interpolate(tensor, size=(hires, hires), mode="bilinear", align_corners=False)
    arr = (tensor.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    Image.fromarray(arr).save(path)


def render_and_composite(renderer, bg_tensor, mvp, resolution=(256, 256), return_clean=False):
    if mvp is None:
        return (bg_tensor, bg_tensor) if return_clean else bg_tensor
    if return_clean:
        adv_rgba, clean_rgba, mask = renderer.render(mvp, resolution=resolution, return_clean=True)
        clean_rgb = clean_rgba.permute(0, 3, 1, 2)
    else:
        adv_rgba, mask = renderer.render(mvp, resolution=resolution)
    adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
    mask_t   = mask.permute(0, 3, 1, 2)
    comp_adv = torch.clamp(adv_rgb * mask_t + bg_tensor * (1 - mask_t), 0.0, 1.0)
    if return_clean:
        comp_clean = torch.clamp(clean_rgb * mask_t + bg_tensor * (1 - mask_t), 0.0, 1.0)
        return comp_adv, comp_clean
    return comp_adv


def inspect_image_structure(processed_inputs, verbose=True):
    img_top_key = None
    for key in ["image", "base_0_rgb", "cam_high", "observation/image"]:
        if key in processed_inputs:
            img_top_key = key
            break
    if img_top_key is None:
        raise RuntimeError(f"找不到图像键！所有键: {list(processed_inputs.keys())}")
    img_val = processed_inputs[img_top_key]
    if isinstance(img_val, dict):
        cam_key = None
        for ck in ["base_0_rgb", "cam_high", "agentview_rgb", "exterior_image_1_left", "image"]:
            if ck in img_val:
                cam_key = ck
                break
        if cam_key is None:
            cam_key = next(iter(img_val))
        img_tensor = img_val[cam_key]
        if verbose:
            print(f"  [图像结构] image 是 dict，包含: {list(img_val.keys())}")
            print(f"  [图像结构] 主相机键: '{cam_key}'，shape={img_tensor.shape}")
    else:
        cam_key = None
        img_tensor = img_val
    with torch.no_grad():
        vmax = img_tensor.float().max().item()
        vmin = img_tensor.float().min().item()
    if vmax > 2.0:
        value_range = "0_255"
    elif vmin < -0.5:
        value_range = "-1_1"
    else:
        value_range = "0_1"
    return img_top_key, cam_key, img_tensor, value_range


def _adapt_adv_image(adv_image_01, orig_tensor, value_range):
    orig_h, orig_w = orig_tensor.shape[-2], orig_tensor.shape[-1]
    if adv_image_01.shape[-2:] != (orig_h, orig_w):
        adv = F.interpolate(adv_image_01, size=(orig_h, orig_w), mode="bilinear", align_corners=False)
    else:
        adv = adv_image_01
    if value_range == "0_255":
        adv = adv * 255.0
    elif value_range == "-1_1":
        adv = adv * 2.0 - 1.0
    adv = adv.to(dtype=orig_tensor.dtype)
    if orig_tensor.dim() == 5:
        T = orig_tensor.shape[1]
        adv = adv.unsqueeze(1).expand(-1, T, -1, -1, -1)
    return adv


def replace_image_in_processed_inputs(processed_inputs, adv_image_01, device, verbose=False):
    img_top_key, cam_key, orig_tensor, value_range = inspect_image_structure(
        processed_inputs, verbose=verbose
    )
    adv_resized = _adapt_adv_image(adv_image_01, orig_tensor, value_range)
    new_inputs  = dict(processed_inputs)
    img_val     = new_inputs[img_top_key]
    if isinstance(img_val, dict):
        new_img_dict = dict(img_val)
        new_img_dict[cam_key] = adv_resized
        new_inputs[img_top_key] = new_img_dict
    else:
        new_inputs[img_top_key] = adv_resized
    return new_inputs, img_top_key, cam_key


def get_processed_inputs_and_observation(policy, raw_inputs_dict):
    import jax
    device = policy._pytorch_device
    inputs = jax.tree.map(lambda x: x, raw_inputs_dict)
    inputs = policy._input_transform(inputs)
    processed_inputs = jax.tree.map(
        lambda x: torch.from_numpy(np.array(x)).to(device).unsqueeze(0),
        inputs
    )
    observation = _model.Observation.from_dict(processed_inputs)
    return processed_inputs, observation, device


def extract_split_features(model, observation):
    images, img_masks, lang_tokens, lang_masks, state = model._preprocess_observation(
        observation, train=False
    )
    img_embs = []
    for img, img_mask in zip(images, img_masks):
        img_emb = model.paligemma_with_expert.embed_image(img)
        img_embs.append(img_emb)
    img_embs = torch.cat(img_embs, dim=1)

    lang_emb = model.paligemma_with_expert.embed_language_tokens(lang_tokens)
    lang_emb = lang_emb * math.sqrt(lang_emb.shape[-1])

    return img_embs, lang_emb, lang_masks


def _extract_first_action(policy_output):
    action = policy_output["actions"] if isinstance(policy_output, dict) else policy_output
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action)
    if action.ndim == 3:
        action = action[0, 0]
    elif action.ndim == 2:
        action = action[0]
    return torch.from_numpy(action.astype(np.float32))


def load_latent_encoder(config_path, ckpt_path, device):
    for taming_root in [Path("./taming-transformers"), Path("./taming-transformers-master")]:
        if taming_root.exists():
            s = str(taming_root.resolve())
            if s not in sys.path:
                sys.path.insert(0, s)
            break

    from taming.models.vqgan import VQModel

    cfg = OmegaConf.load(config_path)
    model_cfg = cfg.model.params if "model" in cfg and "params" in cfg.model else cfg.model
    latent_model = VQModel(**model_cfg)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    latent_model.load_state_dict(state_dict, strict=False)
    latent_model.eval().to(device)
    for p in latent_model.parameters():
        p.requires_grad = False
    return latent_model.encoder


def extract_latent(encoder, img_np, device):
    img = Image.fromarray(img_np).resize((256, 256))
    x = torch.from_numpy(np.asarray(img).astype(np.float32) / 255.0).to(device)
    x = x.permute(2, 0, 1).unsqueeze(0) * 2.0 - 1.0
    with torch.no_grad():
        feat = encoder(x).mean(dim=[2, 3]).squeeze(0)
    return feat.detach()


def compute_frame_weights(latent_features: List[torch.Tensor], tau: float) -> torch.Tensor:
    if len(latent_features) == 0:
        return torch.tensor([], dtype=torch.float32)
    if len(latent_features) == 1:
        return torch.ones(1, dtype=torch.float32)

    feats  = [f.detach().float() for f in latent_features]
    device = feats[0].device
    n      = len(feats)

    v = torch.zeros(n, device=device)
    a = torch.zeros(n, device=device)
    for t in range(1, n):
        v[t] = torch.norm(feats[t] - feats[t - 1], p=2) / 2.0
    for t in range(1, n):
        a[t] = torch.abs(v[t] - v[t - 1])

    def _minmax(x):
        return (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-8)

    s = torch.maximum(_minmax(v), _minmax(a))
    w = torch.softmax(s / float(max(tau, 1e-6)), dim=0)
    return w.detach()


def _gaussian_kernel2d(kernel_size: int, sigma: float, device, dtype):
    radius = kernel_size // 2
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = g / g.sum()
    k2d = torch.outer(g, g)
    return k2d / k2d.sum()


def apply_eot_transforms(img_tensor: torch.Tensor, num_samples: int) -> torch.Tensor:
    if img_tensor.dim() != 4 or img_tensor.shape[0] != 1:
        raise ValueError(f"Expected [1,3,H,W], got {tuple(img_tensor.shape)}")
    device, dtype = img_tensor.device, img_tensor.dtype
    outs = []
    for _ in range(num_samples):
        y = img_tensor
        b = random.uniform(-0.2, 0.2)
        c = random.uniform(0.8, 1.2)
        mean = y.mean(dim=(2, 3), keepdim=True)
        y = (y - mean) * c + mean + b
        k     = random.choice([3, 5])
        sigma = random.uniform(0.5, 1.5)
        kern  = _gaussian_kernel2d(k, sigma, device=device, dtype=dtype)
        kern  = kern.view(1, 1, k, k).repeat(3, 1, 1, 1)
        y = F.conv2d(y, kern, padding=k // 2, groups=3)
        y = torch.clamp(y, 0.0, 1.0)
        outs.append(y.squeeze(0))
    return torch.stack(outs, dim=0)


def perturb_mvp(
    mvp: torch.Tensor,
    rot_std_deg: float = 5.0,
    trans_std: float   = 0.02,
    scale_range: tuple = (0.9, 1.1),
) -> torch.Tensor:
    device, dtype = mvp.device, mvp.dtype
    angles = torch.randn(3, device=device, dtype=dtype) * math.radians(rot_std_deg)
    cx, cy, cz = torch.cos(angles[0]), torch.cos(angles[1]), torch.cos(angles[2])
    sx, sy, sz = torch.sin(angles[0]), torch.sin(angles[1]), torch.sin(angles[2])

    Rx = torch.eye(4, device=device, dtype=dtype)
    Rx[1, 1] = cx;  Rx[1, 2] = -sx
    Rx[2, 1] = sx;  Rx[2, 2] =  cx

    Ry = torch.eye(4, device=device, dtype=dtype)
    Ry[0, 0] = cy;  Ry[0, 2] =  sy
    Ry[2, 0] = -sy; Ry[2, 2] =  cy

    Rz = torch.eye(4, device=device, dtype=dtype)
    Rz[0, 0] = cz;  Rz[0, 1] = -sz
    Rz[1, 0] = sz;  Rz[1, 1] =  cz

    T = torch.eye(4, device=device, dtype=dtype)
    T[:3, 3] = torch.randn(3, device=device, dtype=dtype) * trans_std

    scale = random.uniform(*scale_range)
    S = torch.eye(4, device=device, dtype=dtype)
    S[0, 0] = S[1, 1] = S[2, 2] = scale

    return mvp @ Rz @ Ry @ Rx @ T @ S


# ------------------------------------------------------------------ #
#  核心：直接在图像 tensor 上注入对抗图，保持完整梯度链               #
#  adv_img_01: [1,3,H,W]  float32  [0,1]，有梯度                     #
#  返回带梯度的 img_embs  [B, N_tokens, D]                           #
# ------------------------------------------------------------------ #
def _adv_img_to_embs(policy, adv_img_01: torch.Tensor,
                     processed_inputs: dict, device: str) -> torch.Tensor:
    new_inputs, _, _ = replace_image_in_processed_inputs(
        processed_inputs, adv_img_01, device, verbose=False
    )
    adv_obs = _model.Observation.from_dict(new_inputs)
    # 不加 no_grad，梯度从 adv_img_01 → new_inputs → embed_image → img_embs
    img_embs, _, _ = extract_split_features(policy._model, adv_obs)
    return img_embs


def collect_frame_data(policy, task, task_description, initial_obs_state,
                       num_frames, RENDER_RES, device, cfg, save_dir, episode_idx,
                       search_keywords_list: List[List[str]]):
    model = policy._model

    env, _ = _get_libero_env(task, resolution=RENDER_RES)
    env.reset()
    obs = env.set_init_state(initial_obs_state)
    env.env.sim.forward()

    print(f"[攻击] 收集 {num_frames} 帧数据...")
    latent_encoder = load_latent_encoder(
        cfg.latent_encoder_config, cfg.latent_encoder_ckpt, device
    )

    frame_data   = []
    verbose_done = False

    for t in range(num_frames):
        img_np     = get_libero_image(obs, RENDER_RES)
        raw_inputs = build_raw_inputs(obs, task_description)

        if cfg.save_attack_artifacts and t == 0:
            Image.fromarray(img_np).save(
                os.path.join(save_dir, f"Ep{episode_idx}_Original_F0.png")
            )

        sim    = env.env.sim
        hidden = hide_object_geoms(sim, search_keywords_list)
        sim.forward()
        obs_h   = sim.render(camera_name="agentview", height=RENDER_RES, width=RENDER_RES)
        bg_np   = obs_h[::-1, ::-1].copy()
        restore_object_geoms(sim, hidden)

        bg_tensor = (torch.from_numpy(bg_np).float().to(device) / 255.0
                     ).permute(2, 0, 1).unsqueeze(0)

        model_matrix, body_id, found_name = get_target_model_matrix(env, search_keywords_list)
        if body_id != -1:
            print(f"  [帧 {t}] 找到目标 body: '{found_name}'")
            mvp = build_mvp(env, model_matrix, resolution=(RENDER_RES, RENDER_RES))
        else:
            print(f"  [帧 {t}] 未找到目标 body")
            mvp = None

        with torch.no_grad():
            processed_inputs, observation, _ = get_processed_inputs_and_observation(
                policy, raw_inputs
            )
            clean_img_embs, _, _ = extract_split_features(model, observation)
            clean_img_embs = clean_img_embs.detach()
            clean_action   = _extract_first_action(policy.infer(raw_inputs))
            latent_feat    = extract_latent(latent_encoder, img_np, device)

        if not verbose_done:
            print(f"  [特征] clean_img_embs shape={clean_img_embs.shape}, "
                  f"norm={clean_img_embs.norm():.2f}")
            inspect_image_structure(processed_inputs, verbose=True)
            verbose_done = True

        frame_data.append({
            "bg_tensor":        bg_tensor,
            "mvp":              mvp,
            "processed_inputs": processed_inputs,
            "clean_img_embs":   clean_img_embs,
            "clean_action":     clean_action,
            "latent_feat":      latent_feat,
            "raw_inputs":       raw_inputs,
        })

        obs, _, _, _ = env.step(get_libero_dummy_action())

    return env, frame_data


def train_adversarial_texture_feature_attack(cfg, policy, renderer, initial_obs_state,
                                              task, task_description, save_dir, episode_idx,
                                              search_keywords_list: List[List[str]],
                                              num_iters=50):
    os.makedirs(save_dir, exist_ok=True)
    RENDER_RES = 256
    model      = policy._model
    device     = policy._pytorch_device

    print(f"[攻击] TAAO | 模型: {type(model).__name__}, Device: {device}")
    print(f"[攻击] 攻击目标关键词: {search_keywords_list}")

    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    num_frames = getattr(cfg, "num_frames", 20)
    lambda_nat = getattr(cfg, "lambda_nat", 0.1)

    env, frame_data = collect_frame_data(
        policy, task, task_description, initial_obs_state,
        num_frames, RENDER_RES, device, cfg, save_dir, episode_idx,
        search_keywords_list=search_keywords_list,
    )

    frame_weights = compute_frame_weights(
        [f["latent_feat"] for f in frame_data], tau=cfg.tau
    )
    print(f"[攻击] 关键帧权重: {frame_weights.cpu().numpy().round(4).tolist()}")

    # 梯度诊断：确认 _preprocess_observation 不截断梯度
    _diag_img = torch.zeros(1, 3, RENDER_RES, RENDER_RES,
                            device=device, requires_grad=True)
    _diag_inputs, _, _ = replace_image_in_processed_inputs(
        frame_data[0]["processed_inputs"], _diag_img, device
    )
    _diag_obs  = _model.Observation.from_dict(_diag_inputs)
    _diag_embs, _, _ = extract_split_features(model, _diag_obs)
    print(f"[诊断] img_embs.requires_grad = {_diag_embs.requires_grad}  "
          f"（必须为 True，否则梯度在 _preprocess_observation 内断裂）")
    del _diag_img, _diag_inputs, _diag_obs, _diag_embs

    optimizer = torch.optim.Adam([renderer.get_texture_param()], lr=cfg.attack_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=cfg.attack_lr * 0.1
    )

    loss_history  = []
    grad_log_path = os.path.join(save_dir, f"Ep{episode_idx}_gradient_log.txt")
    with open(grad_log_path, "w") as f:
        f.write("Iter | Total Loss | TAAO Loss | Nat Loss | Grad Norm | LR\n")

    print(f"\n[攻击] 开始优化，共 {num_iters} 轮...")
    iterator = tqdm.tqdm(range(num_iters), desc="TAAO Attack", leave=False)

    for i in iterator:
        optimizer.zero_grad()
        avg_taao = avg_nat = avg_total = 0.0
        valid = 0

        for t, fdata in enumerate(frame_data):
            if fdata["mvp"] is None:
                continue

            # --- 渲染 ---
            adv_diff, clean_diff = render_and_composite(
                renderer, fdata["bg_tensor"], fdata["mvp"],
                resolution=(RENDER_RES, RENDER_RES), return_clean=True,
            )
            nat_loss = F.mse_loss(adv_diff, clean_diff)

            # --- 构造 EoT 样本列表（全程保持梯度） ---
            if cfg.use_eot:
                imgs_for_loss = []
                for _ in range(cfg.eot_num_samples):
                    mvp_p = perturb_mvp(
                        fdata["mvp"],
                        rot_std_deg=cfg.eot_rot_std_deg,
                        trans_std=cfg.eot_trans_std,
                        scale_range=(cfg.eot_scale_min, cfg.eot_scale_max),
                    )
                    adv_rgba_p, mask_p = renderer.render(mvp_p, resolution=(RENDER_RES, RENDER_RES))
                    adv_p = torch.clamp(
                        adv_rgba_p.permute(0, 3, 1, 2) * mask_p.permute(0, 3, 1, 2)
                        + fdata["bg_tensor"] * (1 - mask_p.permute(0, 3, 1, 2)),
                        0.0, 1.0,
                    )
                    # 2D 增强：单次，保持梯度（不调 clone，直接算）
                    b = random.uniform(-0.2, 0.2)
                    c = random.uniform(0.8, 1.2)
                    m = adv_p.mean(dim=(2, 3), keepdim=True)
                    adv_p = torch.clamp((adv_p - m) * c + m + b, 0.0, 1.0)
                    imgs_for_loss.append(adv_p)
            else:
                imgs_for_loss = [adv_diff]

            # --- TAAO loss（梯度从 img_embs 回传到 adv_vc_noise） ---
            taao_sample_losses = []
            for img_k in imgs_for_loss:
                img_embs = _adv_img_to_embs(policy, img_k, fdata["processed_inputs"], device)
                proxy    = img_embs.mean(dim=1)
                clean_proxy = fdata["clean_img_embs"].mean(dim=1).to(device).detach()

                if cfg.attack_mode == "untargeted":
                    taao_k = -F.mse_loss(proxy, clean_proxy)
                elif cfg.attack_mode == "targeted":
                    taao_k = F.mse_loss(proxy, -clean_proxy)
                else:
                    raise ValueError(f"Unsupported attack_mode: {cfg.attack_mode}")
                taao_sample_losses.append(taao_k)

            taao_loss  = torch.stack(taao_sample_losses).mean()
            total_loss = frame_weights[t].to(device) * taao_loss + lambda_nat * nat_loss
            (total_loss / num_frames).backward()

            avg_taao  += taao_loss.item()
            avg_nat   += nat_loss.item()
            avg_total += total_loss.item()
            valid += 1

        if valid > 0:
            avg_taao  /= valid
            avg_nat   /= valid
            avg_total /= valid
        loss_history.append(avg_total)

        grad       = renderer.adv_vc_noise.grad
        g_norm     = grad.norm().item() if grad is not None else 0.0
        current_lr = optimizer.param_groups[0]["lr"]
        with open(grad_log_path, "a") as f:
            f.write(f"{i:02d} | {avg_total:.6f} | {avg_taao:.6f} | {avg_nat:.6f} "
                    f"| {g_norm:.6e} | {current_lr:.6e}\n")

        optimizer.step()
        scheduler.step()
        iterator.set_postfix(TAAO=f"{avg_taao:.4f}", Nat=f"{avg_nat:.4f}", GNorm=f"{g_norm:.4f}")

        if cfg.save_attack_artifacts and (i + 1) % 10 == 0:
            with torch.no_grad():
                for fdata in frame_data:
                    if fdata["mvp"] is not None:
                        save_hires(
                            render_hires(renderer, fdata["bg_tensor"], fdata["mvp"]),
                            os.path.join(save_dir, f"Ep{episode_idx}_AdvRender_iter{i+1:03d}.png"),
                        )
                        break

    # --- 保存产物 ---
    if cfg.save_attack_artifacts:
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            os.path.join(save_dir, f"Ep{episode_idx}_Texture_Noise.pt"),
        )
        Image.fromarray(
            (renderer.get_baked_adv_texture().squeeze().cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(save_dir, f"Ep{episode_idx}_UV_Map.png"))

        with torch.no_grad():
            for fdata in frame_data:
                if fdata["mvp"] is not None:
                    save_hires(
                        render_hires(renderer, fdata["bg_tensor"], fdata["mvp"]),
                        os.path.join(save_dir, f"Ep{episode_idx}_AdvRender_Final.png"),
                    )
                    break

        np.save(os.path.join(save_dir, f"Ep{episode_idx}_loss_history.npy"),
                np.array(loss_history))

        print("\n[诊断] 最终图像特征偏移量:")
        with torch.no_grad():
            for t, fdata in enumerate(frame_data):
                if fdata["mvp"] is not None:
                    adv_final = render_and_composite(
                        renderer, fdata["bg_tensor"], fdata["mvp"],
                        resolution=(RENDER_RES, RENDER_RES),
                    )
                    new_inputs, _, _ = replace_image_in_processed_inputs(
                        fdata["processed_inputs"], adv_final, device
                    )
                    adv_obs = _model.Observation.from_dict(new_inputs)
                    adv_embs, _, _ = extract_split_features(model, adv_obs)
                    delta = (adv_embs - fdata["clean_img_embs"]).norm(dim=-1).mean().item()
                    print(f"  帧 {t}: 图像特征 L2 偏移 = {delta:.4f}")

    return env, loss_history


def evaluate_adversarial_policy(env, policy, renderer, task_description,
                                  initial_state, search_keywords_list: List[List[str]],
                                  max_steps=400):
    print("\n--- 对抗环境 Rollout 评估 ---")
    env.reset()
    obs = env.set_init_state(initial_state)
    sim = env.env.sim
    sim.forward()

    tex_id = find_object_tex_id(sim, search_keywords_list)
    mujoco_tex_injected = False
    tex_restore_info    = None

    if tex_id >= 0:
        baked      = renderer.get_baked_adv_texture()
        adv_tex_np = (baked.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        orig_data, offset, n_bytes = replace_mujoco_texture(sim, tex_id, adv_tex_np)
        tex_restore_info    = (orig_data, offset, n_bytes)
        mujoco_tex_injected = True
        print(f"[评估] MuJoCo 纹理注入成功（tex_id={tex_id}）")
    else:
        print("[评估] ⚠️ 未找到目标纹理 ID，降级为 nvdiffrast 叠加方案")

    device     = policy._pytorch_device
    RENDER_RES = 256
    success    = False
    frames     = []

    try:
        for step in tqdm.tqdm(range(max_steps), desc="评估中"):
            if mujoco_tex_injected:
                img_np  = get_libero_image(obs, RENDER_RES)
                raw_inp = build_raw_inputs(obs, task_description)
            else:
                img_np = get_libero_image(obs, RENDER_RES)
                model_matrix, body_id, _ = get_target_model_matrix(env, search_keywords_list)
                mvp = build_mvp(env, model_matrix, resolution=(RENDER_RES, RENDER_RES)) if body_id != -1 else None
                if mvp is not None:
                    hidden = hide_object_geoms(sim, search_keywords_list)
                    sim.forward()
                    obs_h  = sim.render(camera_name="agentview", height=RENDER_RES, width=RENDER_RES)
                    bg_np  = obs_h[::-1, ::-1].copy()
                    restore_object_geoms(sim, hidden)
                    bg = (torch.from_numpy(bg_np).float().to(device) / 255.0
                          ).permute(2, 0, 1).unsqueeze(0)
                    with torch.no_grad():
                        adv    = render_and_composite(renderer, bg, mvp, resolution=(RENDER_RES, RENDER_RES))
                        img_np = (adv.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                raw_inp = build_raw_inputs(obs, task_description, adv_image_np=img_np)

            frames.append(img_np)
            with torch.no_grad():
                result = policy.infer(raw_inp)
                action = result["actions"] if isinstance(result, dict) else result
                if isinstance(action, torch.Tensor):
                    action = action.cpu().numpy()
                if action.ndim == 3:
                    action = action[0, 0]
                elif action.ndim == 2:
                    action = action[0]

            obs, _, _, _ = env.step(action.tolist())
            if env.env._check_success():
                print(f"\n[结果] 第 {step} 步成功 —— 攻击失败 ❌")
                success = True
                break
    finally:
        if mujoco_tex_injected and tex_restore_info is not None:
            restore_mujoco_texture(sim, tex_id, *tex_restore_info)
            print("[评估] MuJoCo 纹理已恢复")

    if not success:
        print("\n[结果] 达到最大步数，任务失败 —— 攻击成功 ✅")
    return success, frames


@dataclass
class GenerateConfig:
    pretrained_checkpoint: str = "/path/to/checkpoints/pi0_libero_pytorch"

    object_name:           str           = "akita_black_bowl"
    override_mesh_path:    Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path:     Optional[str] = None

    attack_iters: int   = 5000
    attack_lr:    float = 0.01
    num_frames:   int   = 20
    lambda_nat:   float = 0.01

    save_attack_artifacts: bool = True
    local_log_dir:         str  = "./experiments/pi0_attacks"

    latent_encoder_config: str  = "./taming-transformers/configs/vqgan_imagenet_f16_16384.yaml"
    latent_encoder_ckpt:   str  = "./taming-transformers/checkpoints/vqgan_imagenet_f16_16384.ckpt"
    tau:                   float = 1.0

    attack_mode:        str = "untargeted"
    target_action_mode: str = "sign_flip_xyz"

    use_eot:         bool  = False
    eot_num_samples: int   = 4
    eot_rot_std_deg: float = 5.0
    eot_trans_std:   float = 0.02
    eot_scale_min:   float = 0.9
    eot_scale_max:   float = 1.1


@draccus.wrap()
def run_whitebox_attack(cfg: GenerateConfig):
    if cfg.object_name not in OBJECTS:
        raise ValueError(f"未知物体 '{cfg.object_name}'，可选: {list(OBJECTS.keys())}")
    obj_cfg = OBJECTS[cfg.object_name]

    mesh_path    = cfg.override_mesh_path    or obj_cfg["mesh"]
    texture_path = cfg.override_texture_path or obj_cfg["texture"]
    xml_path     = cfg.override_xml_path     or obj_cfg["xml"]
    task_suite   = obj_cfg["task_suite"]
    task_id      = obj_cfg["task_id"]
    search_kw    = obj_cfg["search"]

    print(f"[配置] 物体: {cfg.object_name}")
    print(f"[配置] mesh:    {mesh_path}")
    print(f"[配置] texture: {texture_path}")
    print(f"[配置] xml:     {xml_path}")
    print(f"[配置] task_suite={task_suite}, task_id={task_id}")
    print(f"[配置] 搜索关键词: {search_kw}")

    scale_xyz = parse_mesh_scale(xml_path)
    print(f"[配置] mesh scale: {scale_xyz}")

    print(f"\n加载 Pi0 checkpoint: {cfg.pretrained_checkpoint}")
    config = training_config.get_config("pi0_libero")
    import torch._dynamo
    torch._dynamo.reset()
    torch._dynamo.config.disable = True

    policy = policy_config.create_trained_policy(
        train_config=config,
        checkpoint_dir=cfg.pretrained_checkpoint,
    )
    policy._model.float()
    assert policy._is_pytorch_model, "需要 PyTorch 版本的 checkpoint！"

    device = policy._pytorch_device

    renderer = DifferentiableRenderer(
        mesh_path=mesh_path,
        orig_texture_path=texture_path,
        device=str(device),
        scale_xyz=scale_xyz,
    ).to(device)

    benchmark_dict    = benchmark.get_benchmark_dict()
    task_suite_obj    = benchmark_dict[task_suite]()
    train_task        = task_suite_obj.get_task(task_id)
    train_init_states = task_suite_obj.get_task_init_states(task_id)
    _, train_task_desc = _get_libero_env(train_task, resolution=256)

    DATE_TIME    = time.strftime("%Y%m%d_%H%M%S")
    artifact_dir = os.path.join(cfg.local_log_dir, f"attack_{cfg.object_name}_{DATE_TIME}")

    print(f"\n开始攻击: [{train_task_desc}]")
    env, loss_history = train_adversarial_texture_feature_attack(
        cfg, policy, renderer, train_init_states[0],
        train_task, train_task_desc, artifact_dir, episode_idx=0,
        search_keywords_list=search_kw,
        num_iters=cfg.attack_iters,
    )
    print(f"\n攻击完成，产物保存至: {artifact_dir}")

    is_success, frames = evaluate_adversarial_policy(
        env, policy, renderer, train_task_desc, train_init_states[0],
        search_keywords_list=search_kw,
        max_steps=400,
    )

    if cfg.save_attack_artifacts and frames:
        import imageio
        mp4_path = os.path.join(artifact_dir, "Adversarial_Rollout.mp4")
        writer   = imageio.get_writer(mp4_path, fps=15, codec="libx264", quality=8)
        for frame in frames:
            writer.append_data(frame)
        writer.close()
        print(f"录像保存: {mp4_path}")

    try:
        env.close()
    except Exception:
        pass


if __name__ == "__main__":
    run_whitebox_attack()
    os._exit(0)
