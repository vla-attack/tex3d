import os
import sys
import glob
import ctypes
import traceback
import time
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union, List
import shutil
import xml.etree.ElementTree as ET

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from torch.cuda.amp import autocast
import torchvision
from PIL import Image
from scipy.spatial.transform import Rotation as R
import draccus
import nvdiffrast.torch as dr
import trimesh
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F

print("[INFO] Setting up OSMesa for CPU Rendering...")
os.environ['MUJOCO_GL'] = 'osmesa'
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

try:
    ctypes.CDLL("libOSMesa.so")
except Exception as e:
    print(f"[WARNING] Failed to load libOSMesa.so: {e}")

PATH_TO_LIBERO_ROOT = "/path/to/LIBERO"
ASSET_ROOT_SCANNED  = "/path/to/LIBERO/libero/libero/assets/stable_scanned_objects"
ASSET_ROOT_HOPE     = "/path/to/LIBERO/libero/libero/assets/stable_hope_objects"

if PATH_TO_LIBERO_ROOT not in sys.path:
    sys.path.append(PATH_TO_LIBERO_ROOT)

from libero.libero import benchmark

sys.path.append(str(Path(__file__).parent))
from libero_utils import (
    get_libero_dummy_action, get_libero_env, get_libero_image,
    quat2axisangle, save_rollout_video,
)
sys.path.append(str(Path(__file__).parent.parent))
from openvla_utils import get_processor
from robot_utils import (
    DATE_TIME, get_action, get_image_resize_size, get_model,
    invert_gripper_action, normalize_gripper_action, set_seed_everywhere,
)

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
    """从 XML 中读取 mesh scale，返回 [sx, sy, sz]。"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    for mesh_elem in root.findall(".//mesh"):
        scale_str = mesh_elem.get("scale")
        if scale_str:
            vals = [float(v) for v in scale_str.strip().split()]
            return vals if len(vals) == 3 else [vals[0]] * 3
    return [1.0, 1.0, 1.0]

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
            print(f"[WARNING] Failed to load mesh from {mesh_path}, using dummy box.")
            mesh = trimesh.creation.box(extents=[0.1, 0.1, 0.001])

        vertices = mesh.vertices * scale_arr[None, :]
        self.num_vertices = len(vertices)
        self.register_buffer("pos",   torch.from_numpy(vertices.astype(np.float32)).to(device))
        self.register_buffer("faces", torch.from_numpy(mesh.faces.astype(np.int32)).to(device))

        # UV
        if hasattr(mesh.visual, "uv") and mesh.visual.uv is not None and len(mesh.visual.uv) > 0:
            uv = mesh.visual.uv.astype(np.float32)
            self.register_buffer("uv", torch.from_numpy(uv).to(device))
            if (hasattr(mesh.visual, "face_uv") and mesh.visual.face_uv is not None
                    and len(mesh.visual.face_uv) == len(mesh.faces)):
                self.register_buffer("uv_idx",
                    torch.from_numpy(mesh.visual.face_uv.astype(np.int32)).to(device))
            else:
                self.register_buffer("uv_idx", self.faces)
        else:
            v  = torch.from_numpy(vertices).to(device)
            uv = v[:, :2]
            if uv.numel() > 0:
                mn = uv.min(0, keepdim=True)[0]
                mx = uv.max(0, keepdim=True)[0]
                uv = (uv - mn) / (mx - mn + 1e-8)
            else:
                uv = torch.zeros((len(vertices), 2), device=device)
            self.register_buffer("uv",     uv.float())
            self.register_buffer("uv_idx", self.faces)

        # 顶点法线
        if not hasattr(mesh, "vertex_normals") or mesh.vertex_normals is None:
            mesh.fix_normals()
        vn = np.array(getattr(mesh, "vertex_normals", np.zeros_like(vertices)))
        if vn.sum() == 0:
            vn[:, 2] = 1.0
        vn = vn / (np.linalg.norm(vn, axis=1, keepdims=True) + 1e-8)
        self.register_buffer("vn", torch.from_numpy(vn.astype(np.float32)).to(device))

        self.glctx = dr.RasterizeCudaContext()

        # 原始纹理（带 FLIP_TOP_BOTTOM 以对齐 nvdiffrast UV 坐标系）
        if orig_texture_path and os.path.exists(orig_texture_path):
            img = (Image.open(orig_texture_path).convert("RGB")
                        .resize((self.texture_res, self.texture_res))
                        .transpose(Image.FLIP_TOP_BOTTOM))
            tex = torch.from_numpy(np.array(img)).float() / 255.0
            self.register_buffer("orig_texture",
                tex.unsqueeze(0).to(device).contiguous())
        else:
            fb = (torch.tensor([0.45, 0.45, 0.45], device=device)
                    .view(1, 1, 1, 3)
                    .expand(1, self.texture_res, self.texture_res, 3)
                    .contiguous())
            self.register_buffer("orig_texture", fb)

        self.adv_vc_noise = nn.Parameter(
            torch.zeros((self.num_vertices, 3), dtype=torch.float32, device=device)
        )
        self.light_dir = F.normalize(
            torch.tensor([0.5, 0.5, 1.0], device=device), dim=0
        )

    # -------------------------------------------------------------- #
    def get_texture_param(self):
        return self.adv_vc_noise

    def reset_texture(self):
        with torch.no_grad():
            self.adv_vc_noise.data.fill_(0.0)

    # -------------------------------------------------------------- #
    def render(self, mvp, resolution=(256, 256), return_clean=False):
        pos      = self.pos + self.pos_offset
        pos_homo = torch.cat([pos, torch.ones_like(pos[..., :1])], dim=-1)
        pos_clip = torch.matmul(pos_homo, mvp.t())

        rast, _      = dr.rasterize(self.glctx, pos_clip.unsqueeze(0),
                                    self.faces, resolution=resolution)
        tex_uv, _    = dr.interpolate(self.uv.unsqueeze(0), rast, self.uv_idx)
        clean_color  = dr.texture(self.orig_texture.contiguous(),
                                  tex_uv.contiguous(), filter_mode="linear")

        noise        = torch.tanh(self.adv_vc_noise) * self.epsilon
        noise_interp, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast, self.faces)
        adv_color    = torch.clamp(clean_color + noise_interp, 0.0, 1.0)

        vn_interp, _ = dr.interpolate(self.vn.unsqueeze(0), rast, self.faces)
        vn_interp    = F.normalize(vn_interp, p=2, dim=-1)
        diffuse      = torch.clamp(
            (vn_interp * self.light_dir.view(1, 1, 1, 3)).sum(-1, keepdim=True),
            0.0, 1.0
        )
        lighting = 0.6 + 0.4 * diffuse

        mask = (rast[..., 3] > 0).float().unsqueeze(-1)
        if return_clean:
            return adv_color * lighting, clean_color * lighting, mask
        return adv_color * lighting, mask

    # -------------------------------------------------------------- #
    def get_baked_adv_texture(self):
        """将顶点噪声烘焙回 UV 贴图（导出给 MuJoCo 前翻转回原始方向）。"""
        with torch.no_grad():
            uv_clip = self.uv * 2.0 - 1.0
            uv4     = torch.cat([uv_clip,
                                  torch.zeros_like(uv_clip[..., :1]),
                                  torch.ones_like(uv_clip[..., :1])], dim=-1)
            rast, _ = dr.rasterize(self.glctx, uv4.unsqueeze(0), self.uv_idx,
                                    resolution=(self.texture_res, self.texture_res))
            noise   = torch.tanh(self.adv_vc_noise) * self.epsilon
            baked, _ = dr.interpolate(noise.unsqueeze(0).contiguous(), rast, self.faces)
            mask    = (rast[..., 3] > 0).float().unsqueeze(-1)
            result  = torch.clamp(self.orig_texture + baked * mask, 0.0, 1.0)
            return torch.flip(result, dims=[1])

    # -------------------------------------------------------------- #
    def bake_vertex_colors_to_texture(self, resolution=(256, 256)):
        """兼容旧接口（与 get_baked_adv_texture 等价）。"""
        return self.get_baked_adv_texture()

def get_obj_name(mod, idx, obj_type):
    short_type_map = {
        "texture": "tex", "material": "mat",
        "geom": "geom", "body": "body", "camera": "cam",
    }
    short_type = short_type_map.get(obj_type, obj_type)
    func_name = f"{short_type}_id2name"
    if hasattr(mod, func_name):
        try:
            return getattr(mod, func_name)(idx)
        except Exception:
            pass
    if hasattr(mod, "id2name"):
        try:
            return mod.id2name(idx, obj_type)
        except Exception:
            pass
    return None


def get_target_model_matrix(env, search_keywords_list: List[List[str]]):
    sim = env.unwrapped.sim if hasattr(env, "unwrapped") else env.sim
    target_body_id = -1
    found_name = None

    for keywords in search_keywords_list:
        if hasattr(sim.model, "nbody"):
            for i in range(sim.model.nbody):
                name = get_obj_name(sim.model, i, "body")
                if not name or "vis" in name or "site" in name:
                    continue
                if all(k in name for k in keywords):
                    target_body_id = i
                    found_name = name
                    break
        if target_body_id != -1:
            break

    if target_body_id == -1:
        print(f"[WARNING] Could not find target body for {search_keywords_list}. Using fallback matrix.")
        fallback = torch.eye(4).cuda()
        fallback[2, 3] = 0.85
        return fallback, -1, None

    pos  = sim.data.body_xpos[target_body_id]
    quat = sim.data.body_xquat[target_body_id]
    rot  = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    mat  = np.eye(4, dtype=np.float32)
    mat[:3, :3] = rot.as_matrix()
    mat[:3,  3] = pos
    return torch.from_numpy(mat).cuda(), target_body_id, found_name


def get_render_mvp_from_matrix(env, model_matrix, resolution=(256, 256)):
    sim = env.sim if not hasattr(env, "unwrapped") else env.unwrapped.sim
    W, H = resolution

    cam_id = 0
    try:
        cam_id = sim.model.camera_name2id("agentview")
    except Exception:
        print("[WARNING] Camera 'agentview' not found, using camera 0.")

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


def render_and_composite(renderer, bg_tensor, mvp, resolution=(256, 256)):
    """将对抗渲染结果合成到背景上，返回 [1, 3, H, W]。"""
    if mvp is None:
        return bg_tensor

    adv_rgba, mask = renderer.render(mvp, resolution=resolution)
    adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
    mask_t   = mask.permute(0, 3, 1, 2)
    return torch.clamp(adv_rgb * mask_t + bg_tensor * (1 - mask_t), 0.0, 1.0)


def get_attack_loss(logits, clean_labels):
    """
    Targeted attack loss：将预测推向与正确 bin 对立的方向。
    训练时：minimize CE(predicted, correct_bin)
    攻击时：minimize CE(predicted, opposite_bin)，使动作方向完全反转。
    """
    ACTION_START = 31744  # Llama 词表末尾 256 个 action token 的起始
    ACTION_END   = 32000
    NUM_BINS     = 256

    if logits.shape[1] > clean_labels.shape[1]:
        logits = logits[:, -clean_labels.shape[1]:, :]

    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = clean_labels[:, 1:].contiguous().to(logits.device)

    action_mask = (
        (shift_labels >= ACTION_START) &
        (shift_labels <  ACTION_END)   &
        (shift_labels != -100)
    )
    if not action_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True)

    valid_logits   = shift_logits[action_mask]
    valid_labels   = shift_labels[action_mask]
    action_logits  = valid_logits[:, ACTION_START:ACTION_END]
    correct_bins   = valid_labels - ACTION_START
    opposite_bins  = (NUM_BINS - 1 - correct_bins)

    return F.cross_entropy(action_logits, opposite_bins)


def get_uada_loss_and_metric(logits, labels):
    ACTION_START, ACTION_END, NUM_BINS = 31744, 32000, 256
    if logits.shape[1] > labels.shape[1]:
        logits = logits[:, -labels.shape[1]:, :]
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous().to(logits.device)
    action_mask  = (
        (shift_labels >= ACTION_START) &
        (shift_labels <  ACTION_END)   &
        (shift_labels != -100)
    )
    if not action_mask.any():
        return torch.tensor(0.0, device=logits.device, requires_grad=True), 0.0

    valid_logits  = shift_logits[action_mask]
    valid_labels  = shift_labels[action_mask]
    action_logits = (valid_logits[:, :NUM_BINS]
                     if valid_logits.shape[-1] <= ACTION_END
                     else valid_logits[:, ACTION_START:ACTION_END])

    probs        = F.softmax(action_logits, dim=-1)
    reweigh      = torch.arange(1, NUM_BINS + 1, device=logits.device) / float(NUM_BINS)
    pred_values  = (probs * reweigh).sum(dim=-1)

    target_values = torch.zeros_like(pred_values)
    target_values[valid_labels >  (ACTION_START + NUM_BINS / 2)] = 1.0 / NUM_BINS
    target_values[valid_labels <= (ACTION_START + NUM_BINS / 2)] = 1.0

    return F.mse_loss(pred_values, target_values), 0.0


def train_adversarial_texture(
    cfg, model, processor, renderer,
    initial_obs_state, task, task_description,
    save_dir, episode_idx,
    search_keywords_list: List[List[str]],
    num_iters=20,
):
    print(f"[ATTACK] Training Ep {episode_idx} | Joint Action & Feature Loss "
          f"({cfg.num_frames_to_attack}-Frame Averaged Optimization)...")
    os.makedirs(save_dir, exist_ok=True)

    RENDER_RES       = 256
    model_input_size = get_image_resize_size(cfg)

    env, _ = get_libero_env(task, cfg.model_family, resolution=RENDER_RES)
    env.reset()
    obs = env.set_init_state(initial_obs_state)
    env.env.sim.forward()

    siglip_mean = torch.tensor([0.5,   0.5,   0.5  ], device=model.device).view(1, 3, 1, 1)
    siglip_std  = torch.tensor([0.5,   0.5,   0.5  ], device=model.device).view(1, 3, 1, 1)
    dino_mean   = torch.tensor([0.485, 0.456, 0.406], device=model.device).view(1, 3, 1, 1)
    dino_std    = torch.tensor([0.229, 0.224, 0.225], device=model.device).view(1, 3, 1, 1)

    # ---- 阶段 1：收集多帧环境状态、clean action token、clean hidden states ---- #
    num_frames = cfg.num_frames_to_attack
    frame_data = []
    print(f"[INFO] Collecting {num_frames} frames for averaged optimization...")

    for t in range(num_frames):
        img_np = get_libero_image(obs, RENDER_RES)
        if cfg.save_attack_artifacts and t == 0:
            Image.fromarray(img_np).save(
                os.path.join(save_dir, f"Ep{episode_idx}_Original_F0.png")
            )

        bg_tensor = (torch.from_numpy(img_np).float().to(model.device) / 255.0
                     ).permute(2, 0, 1).unsqueeze(0)  # [1, 3, H, W]

        model_matrix, body_id, found_name = get_target_model_matrix(env, search_keywords_list)
        mvp = (get_render_mvp_from_matrix(env, model_matrix, resolution=(RENDER_RES, RENDER_RES))
               if body_id != -1 else None)
        if body_id != -1:
            print(f"  [帧 {t}] 找到目标 body: '{found_name}'")
        else:
            print(f"  [帧 {t}] 未找到目标 body，跳过该帧渲染")

        # clean action token 预测
        image_pil   = Image.fromarray(img_np).resize((model_input_size, model_input_size))
        prompt      = (f"In: What action should the robot take to "
                       f"{task_description.lower()}?\nOut:")
        clean_inputs = processor(prompt, images=image_pil).to(model.device)
        if "pixel_values" in clean_inputs:
            clean_inputs["pixel_values"] = clean_inputs["pixel_values"].to(torch.bfloat16)

        with torch.no_grad():
            with autocast(dtype=torch.bfloat16):
                clean_output_ids = model.generate(
                    **clean_inputs, max_new_tokens=7,
                    do_sample=False,
                    pad_token_id=processor.tokenizer.pad_token_id,
                )
                # clean hidden states（最后一层）
                clean_224 = F.interpolate(bg_tensor, size=(model_input_size, model_input_size),
                                           mode="bilinear", align_corners=False)
                clean_pv  = torch.cat(
                    [(clean_224 - siglip_mean) / siglip_std,
                     (clean_224 - dino_mean)   / dino_std], dim=1
                ).to(torch.bfloat16)
                clean_fwd = model(
                    input_ids=clean_output_ids,
                    attention_mask=torch.ones_like(clean_output_ids),
                    pixel_values=clean_pv,
                    output_hidden_states=True,
                )
                clean_hidden = clean_fwd.hidden_states[-1].detach()

        frame_data.append({
            "bg_tensor":        bg_tensor,
            "mvp":              mvp,
            "clean_output_ids": clean_output_ids,
            "clean_hidden":     clean_hidden,
        })

        obs, _, _, _ = env.step(get_libero_dummy_action(cfg.model_family))

    # ---- 阶段 2：联合 Loss 优化循环 ---- #
    optimizer = torch.optim.Adam([renderer.get_texture_param()], lr=cfg.attack_lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_iters, eta_min=cfg.attack_lr * 0.1
    )
    loss_history = []

    grad_log_path = os.path.join(save_dir, f"Ep{episode_idx}_gradient_log.txt")
    with open(grad_log_path, "w") as f:
        f.write("Iter | Total Loss | Action Loss | Feature Loss | Grad Norm | LR\n")

    iterator = tqdm.tqdm(range(num_iters), desc="Optimizing", leave=False)
    for i in iterator:
        optimizer.zero_grad()
        avg_total = avg_action = avg_feat = 0.0
        valid = 0

        for fdata in frame_data:
            if fdata["mvp"] is None:
                continue

            bg      = fdata["bg_tensor"]
            mvp     = fdata["mvp"]
            ids     = fdata["clean_output_ids"]
            c_hid   = fdata["clean_hidden"]

            # 可微渲染 + 合成
            adv_rgba, mask = renderer.render(mvp, resolution=(RENDER_RES, RENDER_RES))
            adv_rgb  = adv_rgba.permute(0, 3, 1, 2)
            mask_t   = mask.permute(0, 3, 1, 2)
            adv_diff = torch.clamp(adv_rgb * mask_t + bg * (1 - mask_t), 0.0, 1.0)

            if i == 0 and valid == 0 and cfg.save_attack_artifacts:
                Image.fromarray(
                    (adv_diff.squeeze().permute(1, 2, 0).detach().cpu().numpy() * 255
                     ).astype(np.uint8)
                ).save(os.path.join(save_dir, f"Ep{episode_idx}_Iter0_F0_AdvRender.png"))

            # resize + normalize → model input
            adv_224 = F.interpolate(adv_diff, size=(model_input_size, model_input_size),
                                     mode="bilinear", align_corners=False)
            pv = torch.cat(
                [(adv_224 - siglip_mean) / siglip_std,
                 (adv_224 - dino_mean)   / dino_std], dim=1
            )

            with autocast(dtype=torch.bfloat16):
                outputs = model(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    pixel_values=pv.to(torch.bfloat16),
                    output_hidden_states=True,
                )

            loss_action  = get_attack_loss(outputs.logits, ids)
            # 最大化特征偏移 = 最小化负 MSE
            loss_feature = -F.mse_loss(outputs.hidden_states[-1], c_hid)

            frame_loss = (
                cfg.alpha_action  * loss_action +
                cfg.alpha_feature * loss_feature
            ) / num_frames
            frame_loss.backward()

            avg_total  += frame_loss.item() * num_frames
            avg_action += loss_action.item()
            avg_feat   += loss_feature.item()
            valid += 1

        if valid > 0:
            avg_total  /= valid
            avg_action /= valid
            avg_feat   /= valid
        loss_history.append(avg_total)

        grad   = renderer.adv_vc_noise.grad
        g_norm = grad.norm().item() if grad is not None else 0.0
        cur_lr = optimizer.param_groups[0]["lr"]

        with open(grad_log_path, "a") as f:
            f.write(f"{i:02d} | {avg_total:.6f} | {avg_action:.6f} | "
                    f"{avg_feat:.6f} | {g_norm:.6e} | {cur_lr:.6e}\n")

        optimizer.step()
        scheduler.step()
        iterator.set_postfix(
            act=f"{avg_action:.4f}",
            feat=f"{avg_feat:.4f}",
            gnorm=f"{g_norm:.4f}",
        )

    # ---- 保存产物 ---- #
    if cfg.save_attack_artifacts:
        torch.save(
            renderer.get_texture_param().detach().cpu(),
            os.path.join(save_dir, f"Ep{episode_idx}_Vertex_Noise.pt"),
        )
        final_tex = renderer.get_baked_adv_texture()
        Image.fromarray(
            (final_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        ).save(os.path.join(save_dir, f"Ep{episode_idx}_UV_Map.png"))
        np.save(
            os.path.join(save_dir, f"Ep{episode_idx}_loss_history.npy"),
            np.array(loss_history),
        )

    return env, loss_history


@dataclass
class GenerateConfig:
    model_family:         str            = "openvla"
    pretrained_checkpoint: Union[str, Path] = "/path/to/openvla/checkpoint"
    load_in_8bit:         bool           = False
    load_in_4bit:         bool           = False
    center_crop:          bool           = True
    object_name: str = "akita_black_bowl"
    override_mesh_path:    Optional[str] = None
    override_texture_path: Optional[str] = None
    override_xml_path:     Optional[str] = None
    task_suite_name:    str          = "libero_spatial"
    task_id:            Optional[int] = None  # None = 全部任务
    num_steps_wait:     int          = 10
    num_trials_per_task: int         = 2
    enable_attack:        bool          = False
    attack_iters:         int           = 5000
    attack_lr:            float         = 0.05
    num_frames_to_attack: int           = 20
    alpha_action:         float         = 1.0
    alpha_feature:        float         = 10.0
    save_attack_artifacts: bool         = True
    load_texture_path:     Optional[str] = None
    local_log_dir:         str          = "./experiments/logs"

    use_wandb:     bool = False
    wandb_project: str  = "openvla_attack"
    wandb_entity:  str  = "user"

    seed:       int            = 7
    run_id_note: Optional[str] = None
    unnorm_key:  Optional[str] = None


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> None:
    set_seed_everywhere(cfg.seed)
    if cfg.object_name not in OBJECTS:
        raise ValueError(
            f"未知物体 '{cfg.object_name}'，可选: {list(OBJECTS.keys())}"
        )
    obj_cfg = OBJECTS[cfg.object_name]
    mesh_path    = cfg.override_mesh_path    or obj_cfg["mesh"]
    texture_path = cfg.override_texture_path or obj_cfg["texture"]
    xml_path     = cfg.override_xml_path     or obj_cfg["xml"]
    search_kw    = obj_cfg["search"]
    task_suite_name = cfg.task_suite_name or obj_cfg["task_suite"]

    scale_xyz = parse_mesh_scale(xml_path)

    model     = get_model(cfg)
    processor = get_processor(cfg) if cfg.model_family == "openvla" else None

    renderer = None
    if cfg.enable_attack and cfg.load_texture_path is None:
        print("[INFO] Initializing DifferentiableRenderer...")
        renderer = DifferentiableRenderer(
            mesh_path=mesh_path,
            orig_texture_path=texture_path,
            device=str(model.device),
            scale_xyz=scale_xyz,
        ).to(model.device)

    # ---- 运行 ID / 目录 ---- #
    run_id = f"EVAL-{task_suite_name}-{DATE_TIME}"
    if cfg.run_id_note:
        run_id = f"{cfg.run_id_note}-{run_id}"
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    artifact_dir = os.path.join(cfg.local_log_dir, "attack_artifacts", run_id)
    if cfg.enable_attack and cfg.save_attack_artifacts:
        os.makedirs(artifact_dir, exist_ok=True)

    log_file      = open(os.path.join(cfg.local_log_dir, run_id + ".txt"), "w")
    original_xml  = Path(xml_path)

    if cfg.use_wandb:
        wandb.init(project=cfg.wandb_project, entity=cfg.wandb_entity, name=run_id)

    # 全局纯净 XML 备份
    if not original_xml.exists():
        raise FileNotFoundError(f"XML asset not found at {original_xml}")
    global_clean_backup = original_xml.with_name(
        f"{original_xml.stem}_clean_backup_{DATE_TIME}{original_xml.suffix}"
    )
    shutil.copy(original_xml, global_clean_backup)
    print(f"[INFO] Absolute clean XML backup → {global_clean_backup}")

    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite_obj = benchmark_dict[task_suite_name]()
    target_tasks   = ([cfg.task_id] if cfg.task_id is not None
                      else range(task_suite_obj.n_tasks))

    total_episodes = 0
    total_successes = 0

    try:
        VIDEO_RES = 512

        for task_id in tqdm.tqdm(target_tasks, desc="Tasks"):
            shutil.copy(global_clean_backup, original_xml)
            print(f"\n[INFO] Restored clean XML for Task {task_id}.")

            task = task_suite_obj.get_task(task_id)
            init_states = task_suite_obj.get_task_init_states(task_id)
            if cfg.enable_attack and cfg.load_texture_path is None:
                print(f"[INFO] Attack training for Task {task_id}...")
                renderer.reset_texture()

                dummy_env, train_task_desc = get_libero_env(
                    task, cfg.model_family, resolution=256
                )
                dummy_env.close()

                _, _ = train_adversarial_texture(
                    cfg, model, processor, renderer,
                    init_states[0], task, train_task_desc,
                    artifact_dir, episode_idx=task_id,
                    search_keywords_list=search_kw,
                    num_iters=cfg.attack_iters,
                )
                with torch.no_grad():
                    baked_tex = renderer.get_baked_adv_texture()
                    trained_tex_path = os.path.join(
                        artifact_dir, f"task_{task_id}_adv_texture_{DATE_TIME}.png"
                    )
                    Image.fromarray(
                        (baked_tex.squeeze(0).cpu().numpy() * 255).astype(np.uint8)
                    ).save(trained_tex_path)
                print(f"[INFO] Task {task_id} texture saved → {trained_tex_path}")
                tree = ET.parse(original_xml)
                root = tree.getroot()
                tex_name = f"tex-{cfg.object_name}"
                mat_name = f"mat-{cfg.object_name}"
                for asset_elem in root.findall("asset"):
                    for tex_elem in asset_elem.findall("texture"):
                        if tex_elem.get("name") == tex_name:
                            tex_elem.set("file", str(Path(trained_tex_path).resolve()))
                            tex_elem.set("type", "2d")
                            break
                for mat_elem in root.findall(".//material"):
                    if mat_elem.get("name") == mat_name:
                        mat_elem.set("texuniform", "false")
                tree.write(original_xml)
                print(f"[INFO] XML updated with adversarial texture for Task {task_id}.")
            for ep in tqdm.tqdm(range(cfg.num_trials_per_task),
                                desc=f"Task {task_id} Episodes"):
                env, task_description = get_libero_env(
                    task, cfg.model_family, resolution=VIDEO_RES
                )
                env.reset()
                obs = env.set_init_state(init_states[ep])
                env.env.sim.forward()

                t, max_steps, done, replay_images = 0, 300, False, []

                while t < max_steps + cfg.num_steps_wait:
                    try:
                        if t < cfg.num_steps_wait:
                            obs, _, _, _ = env.step(
                                get_libero_dummy_action(cfg.model_family)
                            )
                            t += 1
                            continue

                        img_high = get_libero_image(obs, VIDEO_RES)
                        replay_images.append(img_high)

                        model_input_size = get_image_resize_size(cfg)
                        img_model = np.array(
                            Image.fromarray(img_high).resize((model_input_size, model_input_size))
                        )

                        observation = {
                            "full_image": img_model,
                            "state": np.concatenate((
                                obs["robot0_eef_pos"],
                                quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            )),
                        }
                        action = get_action(
                            cfg, model, observation, task_description, processor=processor
                        )
                        action = normalize_gripper_action(action, binarize=True)
                        if cfg.model_family == "openvla":
                            action = invert_gripper_action(action)

                        obs, _, done, _ = env.step(action.tolist())
                        if done:
                            total_successes += 1
                            break
                        t += 1
                    except Exception as e:
                        print(f"[ERROR] Task {task_id} Ep {ep} step {t}: {e}")
                        traceback.print_exc()
                        break

                total_episodes += 1
                log_str = f"Task: {task_id} | Ep: {ep} | Success: {done}"
                print(log_str)
                log_file.write(log_str + "\n")
                log_file.flush()

                save_rollout_video(
                    replay_images, total_episodes,
                    success=done, task_description=task_description,
                    log_file=log_file,
                )
                env.close()

        avg_sr = total_successes / total_episodes if total_episodes > 0 else 0.0
        print(f"\n[DONE] Episodes: {total_episodes} | Attack success rate: {avg_sr:.2%}")
        log_file.write(f"\nFINAL AVG SUCCESS RATE: {avg_sr:.2%}\n")

    finally:
        if global_clean_backup.exists():
            shutil.copy(global_clean_backup, original_xml)
            os.remove(global_clean_backup)
            print("[INFO] Original XML restored and backup removed.")
        log_file.close()
        if cfg.use_wandb:
            wandb.finish()


if __name__ == "__main__":
    eval_libero()