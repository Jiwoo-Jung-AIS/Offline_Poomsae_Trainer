# app_gradio_poomsae14.py
from __future__ import annotations
import os, sys, tempfile, platform, math, json, zipfile, time
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HTTP_PROXY", "")
os.environ.setdefault("HTTPS_PROXY", "")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
os.environ.setdefault("GRADIO_LOCALHOST_NAME", "127.0.0.1")

PY_MIN = (3, 9)
if sys.version_info < PY_MIN:
    raise RuntimeError(f"Python>={'.'.join(map(str, PY_MIN))} required, found {platform.python_version()}.")

import torch, os
torch.set_num_threads(2)
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

# Work around Windows Proactor loop quirk
if sys.platform.startswith("win"):
    import asyncio
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def _check_imports():
    missing = []
    try:
        import numpy as np  # noqa
    except Exception:
        missing.append("numpy")
    try:
        import cv2  # noqa
    except Exception:
        missing.append("opencv-python")
    try:
        import gradio  # noqa
    except Exception:
        missing.append("gradio")
    try:
        import scipy  # noqa
        from scipy.linalg import svd  # noqa
    except Exception:
        missing.append("scipy")
    if missing:
        raise RuntimeError(
            "Missing required packages: "
            + ", ".join(missing)
            + "\nInstall with:\n  pip install --upgrade "
            + " ".join(missing)
        )

# replace the unconditional call with:
if os.getenv("POOMSAE_SKIP_IMPORTS") != "1":
    _check_imports()

import numpy as np
import cv2
import gradio as gr

# --- fix gradio-client's schema walker for boolean JSON Schemas ---
try:
    import gradio_client.utils as _gcu

    _orig_json = _gcu._json_schema_to_python_type
    _orig_get_type = _gcu.get_type

    def _get_type_safe(schema):
        # gradio-client expects dict; handle booleans & odd inputs
        if isinstance(schema, bool):
            # True => matches anything; False => matches nothing
            return "boolean" if schema else "never"
        if not isinstance(schema, dict):
            return "any"
        return _orig_get_type(schema)

    def _json_schema_to_python_type_safe(schema, defs):
        # Support the JSON Schema boolean form directly
        if isinstance(schema, bool):
            return "Any" if schema else "Never"
        return _orig_json(schema, defs)

    _gcu.get_type = _get_type_safe
    _gcu._json_schema_to_python_type = _json_schema_to_python_type_safe
except Exception:
    pass
# --- end fix ---


from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Callable, TYPE_CHECKING, Optional
from scipy.linalg import svd
from functools import lru_cache

# ------------------ Pose spec & helpers ------------------

COCO = dict(
    nose=0, l_eye=1, r_eye=2, l_ear=3, r_ear=4,
    l_shoulder=5, r_shoulder=6, l_elbow=7, r_elbow=8,
    l_wrist=9, r_wrist=10, l_hip=11, r_hip=12,
    l_knee=13, r_knee=14, l_ankle=15, r_ankle=16
)
ALIAS_TO_INDEX = {
    'L_foot': COCO['l_ankle'], 'R_foot': COCO['r_ankle'],
    'L_knee': COCO['l_knee'],  'R_knee': COCO['r_knee'],
    'L_elbow': COCO['l_elbow'],'R_elbow': COCO['r_elbow'],
    'L_fist': COCO['l_wrist'], 'R_fist': COCO['r_wrist'],
    'L_shoulder': COCO['l_shoulder'], 'R_shoulder': COCO['r_shoulder']
}
LR_SWAP = {
    'L_foot':'R_foot','R_foot':'L_foot',
    'L_knee':'R_knee','R_knee':'L_knee',
    'L_elbow':'R_elbow','R_elbow':'L_elbow',
    'L_fist':'R_fist','R_fist':'L_fist',
    'L_shoulder':'R_shoulder','R_shoulder':'L_shoulder',
    'hip':'hip','head':'head'
}
SKELETON = [
    (COCO['l_ankle'],COCO['l_knee']),(COCO['l_knee'],COCO['l_hip']),
    (COCO['r_ankle'],COCO['r_knee']),(COCO['r_knee'],COCO['r_hip']),
    (COCO['l_hip'],COCO['r_hip']),
    (COCO['l_shoulder'],COCO['r_shoulder']),
    (COCO['l_hip'],COCO['l_shoulder']),(COCO['r_hip'],COCO['r_shoulder']),
    (COCO['l_shoulder'],COCO['l_elbow']),(COCO['l_elbow'],COCO['l_wrist']),
    (COCO['r_shoulder'],COCO['r_elbow']),(COCO['r_elbow'],COCO['r_wrist']),
    (COCO['nose'],COCO['l_eye']),(COCO['nose'],COCO['r_eye']),
    (COCO['l_eye'],COCO['l_ear']),(COCO['r_eye'],COCO['r_ear']),
    (COCO['nose'],COCO['l_shoulder']),(COCO['nose'],COCO['r_shoulder'])
]

def _safe_float(x, default=0.0):
    try:
        xx = float(x)
        return xx if math.isfinite(xx) else default
    except Exception:
        return default

def _safe_int(x, default=0):
    try:
        xi = int(x)
        return xi
    except Exception:
        return default

def _extract_video_path(v):
    # Accepts gradio VideoData dicts or plain strings
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        if "video" in v and isinstance(v["video"], dict) and "path" in v["video"]:
            return v["video"]["path"]
        if "path" in v:
            return v["path"]
    raise RuntimeError("Could not read video path from the input.")

def _fail(err_msg: str):
    return {"error": err_msg}, None, None

@dataclass
class MovementSpec:
    name: str
    aliases: List[str]
    funcs: Dict[str, Callable[[float], np.ndarray]]
    anchor_alias: str = 'hip'
    retarget_edges: List[Tuple[str,str]] = field(default_factory=list)

# ------------------ Templates for movements ------------------

def right_ap_seogi_funcs():
    def L_foot(t): return np.array([0.59,0.14])
    def L_knee(t): return np.array([0.2074*t+0.4847, 0.0733*t**4-0.1989*t**3+0.129*t**2+0.0071*t+0.546])
    def hip(t):   return np.array([0.3359*t+0.3787, 0.056*t**4-0.1954*t**3+0.1971*t**2-0.0596*t+0.9914])
    def R_knee(t):return np.array([0.587*t+0.1867, -0.0994*t**2+0.1406*t+0.53])
    def R_foot(t):return np.array([0.8438*t-0.0786, 0.3661*t**4-1.0228*t**3+0.7254*t**2-0.0460*t+0.1390])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot}

def right_ap_chagi_funcs():
    def L_foot(t): return np.array([0.6900,0.1300])
    def L_knee(t): return np.array([0.0863*t+0.6558, -0.3704*t**5+1.8446*t**4-3.1465*t**3+2.0026*t**2+0.4830])
    def hip(t):   return np.array([0.2218*t+0.5317, -0.1122*t**2+0.1864*t+0.8955])
    def R_knee(t):return np.array([-0.5926*t**2+1.4319*t+0.3078, 1.1101*t**3-3.9583*t**2+3.7593*t+0.1096])
    def R_foot(t):return np.array([-0.8661*t**2+2.1206*t+0.0426, 1.7849*t**3-6.2892*t**2+5.8082*t-0.4987])
    def head(t):  return np.array([0.2870*t+0.4466, -0.0313*t**2+0.0795*t+1.6666])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot,'head':head}

def right_ap_gubi_funcs():
    def L_foot(t): return np.array([0.9300,0.1200])
    def L_knee(t): return np.array([0.1326*t+0.9841, 0.0227*t**2-0.0759*t+0.5106])
    def hip(t):   return np.array([-0.1191*t**5+0.7457*t**4-1.6581*t**3+1.5200*t**2-0.4806*t+1.5172, 0.4516*t+0.7617])
    def R_knee(t):return np.array([0.7744*t+0.5066, -0.0237*t**3+0.0590*t**2+0.0124*t+0.4620])
    def R_foot(t):return np.array([0.9540*t+0.1779, 0.2929*t**6-2.1359*t**5+5.8894*t**4-7.5048*t**3+4.2331*t**2-0.7730*t+0.1141])
    return {'L_foot':L_foot,'L_knee':L_knee,'hip':hip,'R_knee':R_knee,'R_foot':R_foot}

def right_momtong_jireugi_funcs():
    def L_fist(t):   return np.array([0.8741*t+0.1474, -0.8934*t+1.6191])
    def L_elbow(t):  return np.array([0.1786*t+0.4633, -0.3742*t+1.4563])
    def head(t):     return np.array([0.3800,1.7000])
    def R_elbow(t):  return np.array([-0.4626*t+0.3520, 5.2636*t**5-13.4840*t**4+12.3260*t**3-4.5470*t**2+0.5069*t+1.2876])
    def R_fist(t):   return np.array([-0.5902*t+0.7129, -0.6464*t**3+0.8851*t**2-0.2567*t+1.1724])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'head':head,'R_elbow':R_elbow,'R_fist':R_fist}

def left_arae_makki_funcs():
    def L_fist(t):   return np.array([-0.6249*t+0.9863, 0.2357*t**3-0.5096*t**2+0.1924*t+1.1815])
    def L_elbow(t):  return np.array([-0.4823*t+0.6723, 0.8104*t**5-3.3107*t**4+4.8168*t**3-2.8823*t**2+0.5419*t+1.2583])
    def head(t):     return np.array([0.4100,1.6300])
    def R_elbow(t):  return np.array([0.5103*t+0.1561, -0.1232*t**2+0.3031*t+1.1815])
    def R_fist(t):   return np.array([-0.5699*t+0.3995, -0.2087*t**2+0.3880*t+1.1020])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'head':head,'R_elbow':R_elbow,'R_fist':R_fist}

def right_momtong_makki_funcs():
    def L_fist(t):     return np.array([-1.5982*t+1.3785, -0.2990*t+1.2415])
    def L_elbow(t):    return np.array([-1.4961*t+1.1486, -15.9860*t**4+20.8000*t**3-8.0494*t**2+0.7850*t+1.2542])
    def L_shoulder(t): return np.array([-0.7286*t+0.8559, -25.0090*t**5+38.0250*t**4-19.0200*t**3+3.3308*t**2-0.1087*t+1.3942])
    def head(t):       return np.array([0.6800,1.6300])
    def R_shoulder(t): return np.array([0.2333*t+0.5251, -83.0370*t**6+149.1000*t**5-100.3100*t**4+31.6350*t**3-4.9180*t**2+0.4070*t+1.3910])
    def R_elbow(t):    return np.array([1.1083*t+0.3728, -0.1430*t+1.2878])
    def R_fist(t):     return np.array([1.8585*t+0.2460, -0.4328*t+1.6017])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'L_shoulder':L_shoulder,'head':head,'R_shoulder':R_shoulder,'R_elbow':R_elbow,'R_fist':R_fist}

def left_olgul_makki_funcs():
    def L_fist(t):   return np.array([0.6006*t**3-1.0520*t**2+0.5846*t+0.4677, -2.1389*t**2+2.5386*t+1.1930])
    def L_elbow(t):  return np.array([-3.2700*t**4+5.9761*t**3-3.1766*t**2+0.3222*t+0.5386, -1.5328*t**2+1.7366*t+1.2062])
    def head(t):     return np.array([0.3800,1.6400])
    def R_elbow(t):  return np.array([0.9346*t**2-1.1575*t+0.4457, -1.5860*t**3+2.5606*t**2-1.2124*t+1.3623])
    def R_fist(t):   return np.array([0.5659*t**2-0.7286*t+0.5776, 1.0934*t**2-1.3470*t+1.4880])
    return {'L_fist':L_fist,'L_elbow':L_elbow,'head':head,'R_elbow':R_elbow,'R_fist':R_fist}

MOVEMENTS: Dict[str, MovementSpec] = {
    'right_ap_seogi': MovementSpec('right_ap_seogi',
        ['L_foot','L_knee','hip','R_knee','R_foot'], right_ap_seogi_funcs(), 'hip',
        [('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]),
    'right_ap_chagi': MovementSpec('right_ap_chagi',
        ['L_foot','L_knee','hip','R_knee','R_foot','head'], right_ap_chagi_funcs(), 'hip',
        [('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]),
    'right_ap_gubi': MovementSpec('right_ap_gubi',
        ['L_foot','L_knee','hip','R_knee','R_foot'], right_ap_gubi_funcs(), 'hip',
        [('hip','L_knee'),('L_knee','L_foot'),('hip','R_knee'),('R_knee','R_foot')]),
    'right_momtong_jireugi': MovementSpec('right_momtong_jireugi',
        ['L_fist','L_elbow','head','R_elbow','R_fist'], right_momtong_jireugi_funcs(), 'head', []),
    'left_arae_makki': MovementSpec('left_arae_makki',
        ['L_fist','L_elbow','head','R_elbow','R_fist'], left_arae_makki_funcs(), 'head', []),
    'right_momtong_makki': MovementSpec('right_momtong_makki',
        ['L_fist','L_elbow','L_shoulder','head','R_shoulder','R_elbow','R_fist'],
        right_momtong_makki_funcs(), 'head',
        [('L_shoulder','L_elbow'),('L_elbow','L_fist'),('R_shoulder','R_elbow'),('R_elbow','R_fist')]),
    'left_olgul_makki': MovementSpec('left_olgul_makki',
        ['L_fist','L_elbow','head','R_elbow','R_fist'], left_olgul_makki_funcs(), 'head', [])
}

DISPLAY = {
    'right_ap_seogi': 'Right Ap-seogi',
    'right_ap_chagi': 'Right Ap-chagi',
    'right_ap_gubi': 'Right Ap-gubi',
    'right_momtong_jireugi': 'Right Momtong Jireugi',
    'left_arae_makki': 'Left-hand Arae makki',
    'right_momtong_makki': 'Right-handed Momtong makki',
    'left_olgul_makki': 'Left-handed Olgul makki'
}

# ------------------ Video I/O & math helpers ------------------

def sample_template(spec: MovementSpec, times_01: np.ndarray) -> np.ndarray:
    out = np.zeros((len(times_01), len(spec.aliases), 2), float)
    for i, tt in enumerate(times_01):
        for j, a in enumerate(spec.aliases):
            out[i, j, :] = spec.funcs[a](tt)
    return out

def reflect_template(spec: MovementSpec, X: np.ndarray, times_01: np.ndarray) -> Tuple[List[str], np.ndarray]:
    anchor = np.array([spec.funcs[spec.anchor_alias](tt) for tt in times_01])
    Xmir = X.copy()
    Xmir[...,0] = 2*anchor[:,None,0] - Xmir[...,0]
    aliases_m = [LR_SWAP.get(a,a) for a in spec.aliases]
    order = [aliases_m.index(a) if a in aliases_m else i for i,a in enumerate(spec.aliases)]
    Xmir = Xmir[:, order, :]
    return aliases_m, Xmir

def retarget_limb_lengths(spec: MovementSpec, tpl: np.ndarray, obs: np.ndarray) -> np.ndarray:
    if not spec.retarget_edges:
        return tpl
    out = tpl.copy(); alias_idx = {a: i for i, a in enumerate(spec.aliases)}
    for t in range(tpl.shape[0]):
        for parent, child in spec.retarget_edges:
            if parent not in alias_idx or child not in alias_idx:
                continue
            pi, ci = alias_idx[parent], alias_idx[child]
            v_tpl = out[t, ci] - out[t, pi]
            v_obs = obs[t, ci] - obs[t, pi]
            if not (np.isfinite(v_tpl).all() and np.isfinite(v_obs).all()):
                continue
            len_tpl = float(np.linalg.norm(v_tpl)) + 1e-8
            len_obs = float(np.linalg.norm(v_obs))
            if not np.isfinite(len_obs) or len_obs <= 0:
                continue
            out[t, ci] = out[t, pi] + v_tpl * (len_obs / len_tpl)
    return out

def _read_video_meta(path: str):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("Could not open the uploaded video. Try a standard MP4/H.264 file.")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    N = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return fps, W, H, N

def estimate_stabilization(path: str, stride: int=1):
    cap = cv2.VideoCapture(path)
    prev = None
    transforms: List[np.ndarray] = []
    fail_count = 0
    MAX_FAILS = 5
    max_shift = 50.0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if len(transforms) % max(1, stride) != 0:
            transforms.append(np.eye(3, dtype=np.float32))
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev is None:
            prev = gray
            transforms.append(np.eye(3, dtype=np.float32))
            continue
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            prev_f = prev.astype(np.float32) / 255.0
            gray_f = gray.astype(np.float32) / 255.0
            criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-4)
            _, warp = cv2.findTransformECC(prev_f, gray_f, warp, motionType=cv2.MOTION_AFFINE, criteria=criteria)
            fail_count = 0
            M = np.vstack([warp, [0, 0, 1]]).astype(np.float32)
            M[0,2] = float(np.clip(M[0,2], -max_shift, max_shift))
            M[1,2] = float(np.clip(M[1,2], -max_shift, max_shift))
        except Exception:
            fail_count += 1
            M = np.eye(3, dtype=np.float32)
            if fail_count >= MAX_FAILS:
                transforms.append(M)
                break
        transforms.append(M)
        prev = gray
    cap.release()
    if not transforms:
        return [np.eye(3, dtype=np.float32)]
    cumulative: List[np.ndarray] = []
    C = np.eye(3, dtype=np.float32)
    for M in transforms:
        C = M @ C
        try:
            Cinv = np.linalg.inv(C)
        except np.linalg.LinAlgError:
            Cinv = np.eye(3, dtype=np.float32)
        cumulative.append(Cinv)
    return cumulative

# ------------------ Robust MMPose inferencer ------------------

if TYPE_CHECKING:
    from mmpose.apis import MMPoseInferencer  # noqa

def _have_mmdet_v3():
    try:
        import mmdet  # noqa
        return True
    except Exception:
        return False

def _prepare_mm_shims():
    """Non-invasive shims: do not shadow real installs."""
    import types
    # Ensure mmcv import doesn't require CUDA ops
    sys.modules.pop('mmcv._ext', None)
    try:
        import mmengine.utils.dl_utils.misc as _mmmisc
        _mmmisc.mmcv_full_available = lambda: False
    except Exception:
        pass
    # Complement missing mmcv.ops symbols if needed
    try:
        import mmcv as _mmcv  # type: ignore
        if not hasattr(_mmcv, "__version__"):
            _mmcv.__version__ = "0.0.0"
        if not hasattr(_mmcv, "ops"):
            ops_mod = types.ModuleType('mmcv.ops')
            setattr(_mmcv, 'ops', ops_mod)
            sys.modules['mmcv.ops'] = ops_mod
        if 'mmcv.ops' not in sys.modules:
            sys.modules['mmcv.ops'] = _mmcv.ops  # type: ignore
        ops_mod = sys.modules['mmcv.ops']
        if not hasattr(ops_mod, "MultiScaleDeformableAttention"):
            class _MSDA:  # no-op
                def __init__(self, *args, **kwargs): pass
                def __call__(self, *args, **kwargs): return None
            ops_mod.MultiScaleDeformableAttention = _MSDA  # type: ignore
        if not hasattr(ops_mod, "active_rotated_filter"):
            def active_rotated_filter(*args, **kwargs): return None
            ops_mod.active_rotated_filter = active_rotated_filter  # type: ignore
        if 'mmcv.ops.multi_scale_deform_attn' not in sys.modules:
            msda_mod = types.ModuleType('mmcv.ops.multi_scale_deform_attn')
            msda_mod.MultiScaleDeformableAttention = ops_mod.MultiScaleDeformableAttention  # type: ignore
            sys.modules['mmcv.ops.multi_scale_deform_attn'] = msda_mod
        if 'mmcv.ops.active_rotated_filter' not in sys.modules:
            arf_mod = types.ModuleType('mmcv.ops.active_rotated_filter')
            arf_mod.active_rotated_filter = ops_mod.active_rotated_filter  # type: ignore
            sys.modules['mmcv.ops.active_rotated_filter'] = arf_mod
    except Exception:
        # Create minimal mmcv stub if even import fails (rare on your setup)
        mmcv_pkg = types.ModuleType('mmcv')
        mmcv_pkg.__version__ = "0.0.0"
        sys.modules['mmcv'] = mmcv_pkg
        class _MSDA: 
            def __init__(self, *args, **kwargs): pass
            def __call__(self, *args, **kwargs): return None
        def active_rotated_filter(*args, **kwargs): return None
        ops_mod = types.ModuleType('mmcv.ops')
        ops_mod.MultiScaleDeformableAttention = _MSDA
        ops_mod.active_rotated_filter = active_rotated_filter
        sys.modules['mmcv.ops'] = ops_mod
        setattr(mmcv_pkg, 'ops', ops_mod)
        msda_mod = types.ModuleType('mmcv.ops.multi_scale_deform_attn')
        msda_mod.MultiScaleDeformableAttention = _MSDA
        sys.modules['mmcv.ops.multi_scale_deform_attn'] = msda_mod
        arf_mod = types.ModuleType('mmcv.ops.active_rotated_filter')
        arf_mod.active_rotated_filter = active_rotated_filter
        sys.modules['mmcv.ops.active_rotated_filter'] = arf_mod

@lru_cache(maxsize=1)
def get_inferencer():
    """Prefer true one-stage (detector-free) RTMO or bottom-up. Fallback to top-down with an explicit detector."""
    _prepare_mm_shims()
    from mmpose.apis import MMPoseInferencer

    # Valid detector-free config keys (recognized by MMPose)
    detector_free_configs = [
    "rtmo-s_8xb32-600e_coco-640x640",
    "rtmo-m_16xb16-600e_coco-640x640",
    ]
    last_err = None

    for cfg in detector_free_configs:
        try:
            return MMPoseInferencer(pose2d=cfg, device="cpu", scope="mmpose")
        except Exception as e:
            last_err = e

    # Fallback: top-down (requires a detector model argument!)
    try:
        return MMPoseInferencer(
            pose2d="td-hm_res50_8xb64-210e_coco-256x192",
            det_model="rtmdet-m_8xb32-300e_coco",
            device="cpu",
            scope="mmpose",
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to initialize a detector-free MMPose model.\n"
            f"Last error: {type(last_err).__name__}: {last_err}\n"
            f"Top-down fallback also failed: {type(e).__name__}: {e}\n"
            "Tip: run online once to cache weights, or pre-place them and set MMENGINE_HOME."
        )

# ------------------ Keypoint extraction ------------------

def _pick_best_instance(instances) -> Optional[dict]:
    """
    Given a list of instance dicts from MMPose, select the single person to track.
    Strategy: highest mean keypoint score, tie-break with largest bbox area if present.
    """
    if not instances:
        return None
    best = None
    best_score = -1.0
    best_area = -1.0
    for d in instances:
        # Accept dict-like with keys 'keypoints', 'keypoint_scores', maybe 'bbox'
        kp = np.asarray(d.get('keypoints', [])) if isinstance(d, dict) else None
        if kp is None or kp.size == 0:
            continue
        if 'keypoint_scores' in d:
            sc = np.asarray(d['keypoint_scores'])
        elif kp.shape[-1] >= 3:
            sc = kp[..., 2]
        else:
            sc = np.ones((kp.shape[0],), float) * 0.3
        mean_sc = float(np.nanmean(sc)) if np.isfinite(sc).any() else 0.0
        area = 0.0
        if 'bbox' in d and d['bbox'] is not None and len(d['bbox']) >= 4:
            x1,y1,x2,y2 = d['bbox'][:4]
            area = max(0.0, float(x2-x1)) * max(0.0, float(y2-y1))
        if (mean_sc > best_score) or (abs(mean_sc - best_score) < 1e-6 and area > best_area):
            best = d; best_score = mean_sc; best_area = area
    return best

def extract_keypoints(video_path: str, kpt_thr: float, stride: int=1):
    infer = get_inferencer()
    fps, W, H, N = _read_video_meta(video_path)

    times: List[float] = []
    keyseq: List[np.ndarray] = []
    scoreseq: List[np.ndarray] = []

    try:
        gen = infer(
            video_path,
            return_vis=False,
            kpt_thr=float(kpt_thr),
            draw_bbox=False,
            # Do NOT pass det/bboxes; RTMO/bottom-up handle it internally.
            bbox_thr=0.0
        )
        for i, out in enumerate(gen):
            if i % max(1, stride) != 0:
                continue

            # out['predictions'] can be: [[{...}, {...}, ...]] or [{...}, ...]
            preds = out.get('predictions', [])
            if isinstance(preds, list) and preds and isinstance(preds[0], list):
                persons = preds[0]
            else:
                persons = preds if isinstance(preds, list) else []

            inst = _pick_best_instance(persons)

            xys = np.full((17, 2), np.nan, float)
            scs = np.zeros((17,), float)

            if inst is not None:
                k = np.asarray(inst.get('keypoints', []))
                if k.size:
                    if k.shape[-1] >= 2:
                        xys = k[..., :2]
                    if 'keypoint_scores' in inst:
                        scs = np.asarray(inst['keypoint_scores'])
                    elif k.shape[-1] >= 3:
                        scs = k[..., 2]

                # If model uses different key order/size, try best-effort map to COCO-17 (truncate/pad)
                if xys.shape[0] != 17:
                    # Simple heuristic: take first 17 or pad with NaN
                    if xys.shape[0] > 17:
                        xys = xys[:17]
                        scs = scs[:17]
                    else:
                        pad = 17 - xys.shape[0]
                        xys = np.vstack([xys, np.full((pad, 2), np.nan, float)])
                        scs = np.hstack([scs, np.zeros((pad,), float)])

            times.append(i / max(1.0, fps))
            keyseq.append(xys)
            scoreseq.append(scs)
    except Exception as e:
        raise RuntimeError(f"Pose inference failed: {e}")

    return np.array(times), np.array(keyseq), np.array(scoreseq)

# ------------------ Geometry & scoring ------------------

def weighted_similarity_transform(A: np.ndarray, B: np.ndarray, w: np.ndarray):
    valid = np.isfinite(A).all(axis=1) & np.isfinite(B).all(axis=1) & np.isfinite(w) & (w > 0)
    if valid.sum() < 2:
        return 1.0, np.eye(2), np.zeros(2)
    A = A[valid]; B = B[valid]; w = w[valid]
    w = w / (w.sum() + 1e-8)
    muA = (w[:, None] * A).sum(0)
    muB = (w[:, None] * B).sum(0)
    A0 = A - muA
    B0 = B - muB
    H = (A0 * w[:, None]).T @ B0
    U, S, Vt = svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[1, :] *= -1
        R = Vt.T @ U.T
    denom = (w[:, None] * (A0 ** 2)).sum()
    s = float(S.sum() / (denom + 1e-8))
    t = muB - s * (R @ muA)
    return s, R, t

def frame_error(template_J2: np.ndarray, observed_J2: np.ndarray, conf_J: np.ndarray):
    conf = np.where(np.isfinite(conf_J), conf_J, 0.0)
    s, R, t = weighted_similarity_transform(template_J2, observed_J2, conf)
    mapped = (s * (template_J2 @ R.T)) + t
    d = np.linalg.norm(mapped - observed_J2, axis=1)
    valid = np.isfinite(d) & np.isfinite(conf) & (conf > 0)
    if valid.sum() == 0:
        return 1e9
    rmse = float(np.sqrt((conf[valid] * (d[valid] ** 2)).sum() / (conf[valid].sum() + 1e-8)))
    return rmse

def eval_segment(P_seg: np.ndarray, W_seg: np.ndarray, spec: MovementSpec):
    T = P_seg.shape[0]
    t01 = np.linspace(0, 1, max(T, 2))
    tpl = sample_template(spec, t01[:T])
    _, tplm = reflect_template(spec, tpl, t01[:T])
    tpl_rt  = retarget_limb_lengths(spec, tpl,  P_seg)
    tplm_rt = retarget_limb_lengths(spec, tplm, P_seg)
    errs_o, errs_m = [], []
    for i in range(T):
        if not (np.isfinite(P_seg[i]).any() and np.isfinite(W_seg[i]).any()):
            continue
        errs_o.append(frame_error(tpl_rt[i],  P_seg[i], W_seg[i]))
        errs_m.append(frame_error(tplm_rt[i], P_seg[i], W_seg[i]))
    if not errs_o:
        return 1e9, 'right'
    err_o = float(np.nanmean(errs_o))
    err_m = float(np.nanmean(errs_m))
    return (err_m, 'left') if err_m < err_o else (err_o, 'right')

def stabilize_keypoints(keyseq: np.ndarray, cumulative_transforms: List[np.ndarray]):
    out = keyseq.copy()
    L = min(len(out), len(cumulative_transforms))
    for i in range(L):
        M = cumulative_transforms[i]
        for j in range(out.shape[1]):
            x, y = out[i, j]
            if not np.isfinite(x) or not np.isfinite(y):
                continue
            v = np.array([x, y, 1.0], dtype=np.float32)
            vv = M @ v
            out[i, j, 0] = float(vv[0])
            out[i, j, 1] = float(vv[1])
    return out

def _draw_pose(img: np.ndarray, kpts: np.ndarray, scores: np.ndarray, thr: float=0.2):
    if img is None or not img.size:
        return
    for a,b in SKELETON:
        if a < len(kpts) and b < len(kpts):
            xa,ya = kpts[a]; xb,yb = kpts[b]
            wa = scores[a] if a < len(scores) else 0.0
            wb = scores[b] if b < len(scores) else 0.0
            if all(map(np.isfinite,[xa,ya,xb,yb])) and wa>=thr and wb>=thr:
                cv2.line(img, (int(xa),int(ya)), (int(xb),int(yb)), (0,255,0), 2)
    for i,(x,y) in enumerate(kpts):
        if np.isfinite(x) and np.isfinite(y) and (i < len(scores) and scores[i] >= thr):
            cv2.circle(img, (int(x),int(y)), 3, (255,0,0), -1)

def save_overlay_zip(video_path: str,
                     keys: np.ndarray,
                     scores: np.ndarray,
                     times: np.ndarray,
                     cumulative_transforms: Optional[List[np.ndarray]],
                     max_frames: int = 60) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Failed to reopen video for overlays.")
    total = len(keys)
    if total == 0:
        cap.release()
        raise RuntimeError("No keypoints available for overlay output.")
    take = min(max_frames, total)
    idxs = np.linspace(0, total-1, take).astype(int)
    tmpdir = tempfile.mkdtemp(prefix="poomsae_overlay_")
    zip_path = os.path.join(tmpdir, "overlays.zip")
    zf = zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED)
    fps, W, H, N = _read_video_meta(video_path)
    frame_numbers = np.clip(np.round(times * fps).astype(int), 0, max(0, N-1))
    wanted = set(frame_numbers[idxs].tolist())
    current = 0
    ok, frame = cap.read()
    written = 0
    while ok:
        if current in wanted:
            nearest_idx = int(np.argmin(np.abs(frame_numbers - current)))
            k = keys[nearest_idx]
            s = scores[nearest_idx]
            if cumulative_transforms and nearest_idx < len(cumulative_transforms):
                pass
            img = frame.copy()
            _draw_pose(img, k, s, thr=0.2)
            ok_png, buf = cv2.imencode(".png", img)
            if ok_png:
                zf.writestr(f"frame_{current:06d}.png", buf.tobytes())
                written += 1
            if written >= take:
                break
        current += 1
        ok, frame = cap.read()
    zf.close()
    cap.release()
    if written == 0:
        raise RuntimeError("Overlay export produced 0 frames (video too short or no valid keypoints).")
    return zip_path

# ------------------ Pipeline ------------------

def analyze_poomsae(video,
                    segments:int,
                    kpt_thr:float,
                    stabilize:bool,
                    stride:int,
                    min_confidence:float,
                    save_overlays:bool):
    try:
        if video is None:
            return _fail("Please upload a video.")
        path = _extract_video_path(video)
        fps, W, H, N = _read_video_meta(path)
        times, keys, scores = extract_keypoints(path, kpt_thr=kpt_thr, stride=max(1, int(stride)))
        if times.size == 0 or keys.size == 0:
            return _fail("No frames/keypoints extracted. Try lowering the keypoint threshold or using a clearer video.")
        cumulative = None
        if bool(stabilize):
            cumulative = estimate_stabilization(path, stride=max(1, int(stride)))
            keys = stabilize_keypoints(keys, cumulative)

        S = max(1, int(segments))
        splits = np.linspace(0, keys.shape[0], S+1).astype(int)
        per_segment = []
        for s in range(S):
            a,b = splits[s], splits[s+1]
            if b <= a+1:
                per_segment.append({
                    "segment": s+1,
                    "start": float(times[a]),
                    "end": float(times[min(b-1, a)]),
                    "best_movement": None, "side": None,
                    "score": 0.0, "rmse": None,
                    "note": "Too few frames in this segment"
                })
                continue
            P = keys[a:b]
            Wc = np.clip(scores[a:b], 0, 1)
            seg_conf = float(np.nanmean(Wc[np.isfinite(Wc)])) if np.isfinite(Wc).any() else 0.0
            if seg_conf < float(min_confidence):
                per_segment.append({
                    "segment": s+1,
                    "start": float(times[a]),
                    "end": float(times[b-1]),
                    "best_movement": None,
                    "side": None,
                    "score": 0.0,
                    "rmse": None,
                    "note": f"Skipped (mean confidence {seg_conf:.3f} < {float(min_confidence):.3f})"
                })
                continue
            results = []
            for kname, spec in MOVEMENTS.items():
                obs = np.zeros((P.shape[0], len(spec.aliases), 2), float)
                conf = np.zeros((P.shape[0], len(spec.aliases)), float)
                for j,a_name in enumerate(spec.aliases):
                    if a_name == 'hip':
                        obs[:, j, :] = (P[:, COCO['l_hip'], :] + P[:, COCO['r_hip'], :]) / 2.0
                        conf[:, j]    = np.minimum(Wc[:, COCO['l_hip']], Wc[:, COCO['r_hip']])
                    elif a_name == 'head':
                        face_idxs = [COCO['nose'], COCO['l_eye'], COCO['r_eye'], COCO['l_ear'], COCO['r_ear']]
                        head = np.nanmean(P[:, face_idxs, :], axis=1)
                        obs[:, j, :] = head
                        finite = np.isfinite(P[:, face_idxs, :]).all(axis=2)
                        num_finite = finite.sum(axis=1)
                        frac = num_finite / len(face_idxs)
                        mean_scores = np.where(num_finite > 0, Wc[:, face_idxs].mean(axis=1), 0.0)
                        conf[:, j] = frac * mean_scores
                    else:
                        idx = ALIAS_TO_INDEX[a_name]
                        obs[:, j, :] = P[:, idx, :]
                        conf[:, j]    = Wc[:, idx]
                rmse, side = eval_segment(obs, conf, spec)
                results.append((kname, rmse, side))
            results.sort(key=lambda x: x[1])
            best_k, best_rmse, side = results[0]
            score = float(np.clip(1.0 - best_rmse/0.15, 0.0, 1.0))
            per_segment.append({
                "segment": s+1,
                "start": float(times[a]),
                "end": float(times[b-1]),
                "best_movement": DISPLAY.get(best_k, best_k),
                "side": side,
                "score": score,
                "rmse": float(best_rmse)
            })
        csv_lines = ["segment,start,end,best_movement,side,score,rmse,note"]
        for it in per_segment:
            csv_lines.append(
                f"{it['segment']},{it['start']:.3f},{it['end']:.3f},"
                f"{it.get('best_movement')},{it.get('side')},"
                f"{_safe_float(it.get('score'),0.0):.3f},"
                f"{_safe_float(it.get('rmse'),float('nan')) if it.get('rmse') is not None else ''},"
                f"{json.dumps(it.get('note',''))}"
            )
        csv_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        with open(csv_path,'w',encoding='utf-8') as f:
            f.write("\n".join(csv_lines))
        summary = {
            "fps": float(fps),
            "resolution": f"{int(W)}x{int(H)}",
            "frames": int(N),
            "stride": int(max(1, int(stride))),
            "segments_requested": int(S),
            "min_confidence": float(min_confidence),
            "stabilized": bool(stabilize)
        }
        zip_path = None
        if bool(save_overlays):
            try:
                zip_path = save_overlay_zip(
                    video_path=path,
                    keys=keys,
                    scores=scores,
                    times=times,
                    cumulative_transforms=cumulative,
                    max_frames=60
                )
            except Exception as e:
                summary["overlay_warning"] = f"Overlay export failed: {e}"
        result_json = {"summary": summary, "segments": per_segment}
        return result_json, csv_path, (zip_path if zip_path else None)
    except Exception as e:
        msg = f"{e.__class__.__name__}: {e}"
        return _fail(f"Processing failed safely. Details: {msg}")

def analyze_poomsae_stream(v, segs, thr, stab, strd, mconf, overlays):
    try:
        if v is None:
            yield json.dumps({"error": "Please upload a video."}, ensure_ascii=False), None, None
            return

        path = _extract_video_path(v)
        fps, W, H, N = _read_video_meta(path)
        stride_i = max(1, _safe_int(strd, 1))
        total_est = max(1, int(N // stride_i))
        start = time.perf_counter()

        # Phase 1 ??pose inference with streaming
        yield json.dumps({"phase": "pose_inference", "status": "starting",
                          "fps": float(fps), "resolution": f"{int(W)}x{int(H)}",
                          "frames": int(N), "stride": stride_i}, ensure_ascii=False), None, None

        infer = get_inferencer()
        times, keyseq, scoreseq = [], [], []
        processed = 0

        gen = infer(
            path,
            return_vis=False,
            kpt_thr=float(_safe_float(thr, 0.30)),
            draw_bbox=False,
            bbox_thr=0.0
        )

        for i, out in enumerate(gen):
            if i % stride_i != 0:
                continue

            # Decode predictions (same as extract_keypoints, inlined so we can yield)
            preds = out.get('predictions', [])
            if isinstance(preds, list) and preds and isinstance(preds[0], list):
                persons = preds[0]
            else:
                persons = preds if isinstance(preds, list) else []

            inst = _pick_best_instance(persons)

            xys = np.full((17, 2), np.nan, float)
            scs = np.zeros((17,), float)

            if inst is not None:
                k = np.asarray(inst.get('keypoints', []))
                if k.size:
                    if k.shape[-1] >= 2:
                        xys = k[..., :2]
                    if 'keypoint_scores' in inst:
                        scs = np.asarray(inst['keypoint_scores'])
                    elif k.shape[-1] >= 3:
                        scs = k[..., 2]

                if xys.shape[0] != 17:
                    if xys.shape[0] > 17:
                        xys = xys[:17]; scs = scs[:17]
                    else:
                        pad = 17 - xys.shape[0]
                        xys = np.vstack([xys, np.full((pad, 2), np.nan, float)])
                        scs = np.hstack([scs, np.zeros((pad,), float)])

            times.append(i / max(1.0, float(fps)))
            keyseq.append(xys)
            scoreseq.append(scs)
            processed += 1

            # Stream progress every ~10 processed frames
            if processed % 10 == 0 or processed == total_est:
                elapsed = time.perf_counter() - start
                cur_fps = processed / elapsed if elapsed > 0 else 0.0
                eta = int((total_est - processed) / cur_fps) if cur_fps > 0 else None
                yield json.dumps({
                    "phase": "pose_inference",
                    "frames_processed": processed,
                    "total_frames_est": total_est,
                    "progress_pct": round(100 * processed / total_est, 2),
                    "fps_runtime": round(cur_fps, 2),
                    "eta_sec": eta
                }, ensure_ascii=False), None, None

        times = np.array(times); keys = np.array(keyseq); scores = np.array(scoreseq)
        if times.size == 0 or keys.size == 0:
            yield json.dumps({"error": "No frames/keypoints extracted. Try a clearer video or lower keypoint threshold."}, ensure_ascii=False), None, None
            return

        # Phase 2 ??optional stabilization
        if bool(stab):
            yield json.dumps({"phase": "stabilization", "status": "running"}, ensure_ascii=False), None, None
            cumulative = estimate_stabilization(path, stride=stride_i)
            keys = stabilize_keypoints(keys, cumulative)
        else:
            cumulative = None

        # Phase 3 ??scoring per segment with streaming
        S = max(1, _safe_int(segs, 14))
        splits = np.linspace(0, keys.shape[0], S + 1).astype(int)
        per_segment = []
        for s in range(S):
            a, b = splits[s], splits[s + 1]

            phase_msg = {"phase": "scoring", "segment_index": s + 1, "segments_total": S,
                         "progress_pct": round(100 * (s) / S, 2)}
            yield json.dumps(phase_msg, ensure_ascii=False), None, None

            if b <= a + 1:
                per_segment.append({
                    "segment": s + 1, "start": float(times[a]),
                    "end": float(times[min(b - 1, a)]),
                    "best_movement": None, "side": None, "score": 0.0, "rmse": None,
                    "note": "Too few frames in this segment"
                })
                continue

            P = keys[a:b]
            Wc = np.clip(scores[a:b], 0, 1)
            seg_conf = float(np.nanmean(Wc[np.isfinite(Wc)])) if np.isfinite(Wc).any() else 0.0

            if seg_conf < float(_safe_float(mconf, 0.15)):
                per_segment.append({
                    "segment": s + 1, "start": float(times[a]), "end": float(times[b - 1]),
                    "best_movement": None, "side": None, "score": 0.0, "rmse": None,
                    "note": f"Skipped (mean confidence {seg_conf:.3f} < {float(_safe_float(mconf, 0.15)):.3f})"
                })
                continue

            results = []
            for kname, spec in MOVEMENTS.items():
                obs = np.zeros((P.shape[0], len(spec.aliases), 2), float)
                conf = np.zeros((P.shape[0], len(spec.aliases)), float)

                for j, a_name in enumerate(spec.aliases):
                    if a_name == 'hip':
                        obs[:, j, :] = (P[:, COCO['l_hip'], :] + P[:, COCO['r_hip'], :]) / 2.0
                        conf[:, j] = np.minimum(Wc[:, COCO['l_hip']], Wc[:, COCO['r_hip']])
                    elif a_name == 'head':
                        face_idxs = [COCO['nose'], COCO['l_eye'], COCO['r_eye'], COCO['l_ear'], COCO['r_ear']]
                        head = np.nanmean(P[:, face_idxs, :], axis=1)
                        obs[:, j, :] = head
                        finite = np.isfinite(P[:, face_idxs, :]).all(axis=2)
                        num_finite = finite.sum(axis=1)
                        frac = num_finite / len(face_idxs)
                        mean_scores = np.where(num_finite > 0, Wc[:, face_idxs].mean(axis=1), 0.0)
                        conf[:, j] = frac * mean_scores
                    else:
                        idx = ALIAS_TO_INDEX[a_name]
                        obs[:, j, :] = P[:, idx, :]
                        conf[:, j] = Wc[:, idx]

                rmse, side = eval_segment(obs, conf, spec)
                results.append((kname, rmse, side))

            results.sort(key=lambda x: x[1])
            best_k, best_rmse, side = results[0]
            score = float(np.clip(1.0 - best_rmse / 0.15, 0.0, 1.0))
            per_segment.append({
                "segment": s + 1,
                "start": float(times[a]),
                "end": float(times[b - 1]),
                "best_movement": DISPLAY.get(best_k, best_k),
                "side": side,
                "score": score,
                "rmse": float(best_rmse)
            })

        # Build outputs (CSV, optional ZIP)
        csv_lines = ["segment,start,end,best_movement,side,score,rmse,note"]
        for it in per_segment:
            csv_lines.append(
                f"{it['segment']},{it['start']:.3f},{it['end']:.3f},"
                f"{it.get('best_movement')},{it.get('side')},"
                f"{_safe_float(it.get('score'),0.0):.3f},"
                f"{_safe_float(it.get('rmse'),float('nan')) if it.get('rmse') is not None else ''},"
                f"{json.dumps(it.get('note',''))}"
            )
        csv_path = tempfile.NamedTemporaryFile(delete=False, suffix='.csv').name
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("\n".join(csv_lines))

        summary = {
            "fps": float(fps),
            "resolution": f"{int(W)}x{int(H)}",
            "frames": int(N),
            "stride": stride_i,
            "segments_requested": int(max(1, _safe_int(segs, 14))),
            "min_confidence": float(_safe_float(mconf, 0.15)),
            "stabilized": bool(stab)
        }

        zip_path = None
        if bool(overlays):
            yield json.dumps({"phase": "overlays", "status": "exporting"}, ensure_ascii=False), None, None
            try:
                zip_path = save_overlay_zip(
                    video_path=path,
                    keys=keys,
                    scores=scores,
                    times=times,
                    cumulative_transforms=cumulative,
                    max_frames=60
                )
            except Exception as e:
                summary["overlay_warning"] = f"Overlay export failed: {e}"

        result_json = {"summary": summary, "segments": per_segment}
        yield json.dumps(result_json, indent=2, ensure_ascii=False), csv_path, (zip_path if zip_path else None)

    except Exception as e:
        yield json.dumps({"error": f"Processing failed safely. Details: {e.__class__.__name__}: {e}"}, ensure_ascii=False), None, None

# ------------------ Gradio UI ------------------

with gr.Blocks(title="Poomsae 14 Analyzer", theme="default") as demo:
    gr.Markdown("## Advanced Poomsae Trainer (14 segments) ??Offline & Robust")
    with gr.Row():
        vid = gr.Video(label="Poomsae video (MP4/H.264 recommended)", sources=["upload"], include_audio=False)
    with gr.Row():
        segments = gr.Slider(1, 20, value=14, step=1, label="Number of segments")
        kpt_thr = gr.Slider(0.0, 1.0, value=0.30, step=0.05, label="Keypoint threshold")
        stride = gr.Slider(1, 5, value=1, step=1, label="Sample every Nth frame")
    with gr.Row():
        stabilize = gr.Checkbox(value=True, label="Stabilize camera (robust ECC)")
        min_conf = gr.Slider(0.0, 1.0, value=0.15, step=0.05, label="Min mean confidence per segment (skip below)")
        save_overlays = gr.Checkbox(value=False, label="Save overlays (ZIP of PNG frames)")
    go = gr.Button("Analyze", variant="primary")
    out_json = gr.Textbox(label="Results (JSON)", lines=12, autoscroll=True)
    out_csv = gr.File(label="Download CSV")
    out_zip = gr.File(label="Overlays ZIP (optional)")

    def _run(v, segs, thr, stab, strd, mconf, overlays):
        res_json, csv_path, zip_path = analyze_poomsae(
            v,
            _safe_int(segs, 14),
            _safe_float(thr, 0.30),
            bool(stab),
            _safe_int(strd, 1),
            _safe_float(mconf, 0.15),
            bool(overlays),
        )
        json_str = json.dumps(res_json, indent=2, ensure_ascii=False)
        return json_str, csv_path, zip_path

    go.click(_run,              # <??was _run
        [vid, segments, kpt_thr, stabilize, stride, min_conf, save_overlays],
        [out_json, out_csv, out_zip],
        show_progress="full",
        api_name="analyze",
        concurrency_limit=1,                 # <??add this
    )   



if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        inbrowser=False,
        show_api=True,     # show docs; also proves API is mounted
        share=False,
        max_threads=1,
    )

