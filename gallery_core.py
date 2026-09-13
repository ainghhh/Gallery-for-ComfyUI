# -*- coding: utf-8 -*-
"""
Gallery4ComfyUI · 核心引擎（发布版）
================================
- 双数据源：ComfyUI output / Stable Diffusion WebUI outputs
- PNG metadata 解析（ComfyUI prompt/workflow + SD WebUI parameters）
- 增量索引缓存、搜索/筛选/排序/分页、模型统计、收藏
- 首次运行需配置 SD WebUI 根目录（存于插件 userdata/settings.json）
零第三方依赖（ComfyUI 环境自带 aiohttp / folder_paths）。
"""
import os, json, time, zlib, threading, re, random

try:
    import folder_paths
except Exception:
    folder_paths = None

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
USERDATA = os.path.join(PLUGIN_DIR, "userdata")
SETTINGS_FILE = os.path.join(USERDATA, "settings.json")
INDEX_DIR = os.path.join(USERDATA, "index")
DATA_DIR = os.path.join(PLUGIN_DIR, "data")   # 可选数据目录（画师库/WiLin备注，发布版不含）


def _data_file(name):
    """数据文件查找：环境变量 GALLERY4_DATA_DIR 优先，其次插件目录 data/。不存在返回 ""。"""
    env_dir = os.environ.get("GALLERY4_DATA_DIR", "").strip()
    for d in (env_dir, DATA_DIR):
        if not d:
            continue
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return ""

_indexes = {}          # source -> {entries: [...], ready: bool, total: int, scanned: int}
_indexes_lock = threading.Lock()
_favs = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")
VIDEO_EXTS = (".mp4", ".webm", ".mov", ".mkv", ".avi", ".gif")


# ---------------------------------------------------------------------------
# 设置（SD WebUI 根目录）
# ---------------------------------------------------------------------------
def load_settings():
    try:
        with open(SETTINGS_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_settings(webui_root):
    root = (webui_root or "").strip()
    if root and not os.path.isdir(root):
        return {"ok": False, "error": "目录不存在: %s" % root}
    os.makedirs(USERDATA, exist_ok=True)
    with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
        json.dump({"webui_root": root}, f, ensure_ascii=False, indent=1)
    if root:
        # 配置变化后触发后台重建 WebUI 索引
        start_scan("webui", force=True)
    return {"ok": True, "webui_root": root}

def webui_outputs_dir():
    root = load_settings().get("webui_root", "")
    if root and os.path.isdir(os.path.join(root, "outputs")):
        return os.path.join(root, "outputs")
    return None

def webui_models_dir():
    root = load_settings().get("webui_root", "")
    p = os.path.join(root, "models", "Stable-diffusion") if root else ""
    return p if os.path.isdir(p) else None


# ---------------------------------------------------------------------------
# PNG metadata 解析
# ---------------------------------------------------------------------------
def read_png_texts(path):
    texts = {}
    try:
        with open(path, "rb") as f:
            if f.read(8) != b"\x89PNG\r\n\x1a\n":
                return texts
            while True:
                hdr = f.read(8)
                if len(hdr) < 8:
                    break
                length = int.from_bytes(hdr[:4], "big")
                ctype = hdr[4:8]
                data = f.read(length)
                f.read(4)
                if ctype == b"tEXt":
                    kw, _, val = data.partition(b"\x00")
                    texts[kw.decode("latin-1")] = val.decode("utf-8", "replace")
                elif ctype == b"iTXt":
                    kw, _, rest = data.partition(b"\x00")
                    if len(rest) >= 2:
                        comp = rest[0]
                        rest2 = rest[2:]
                        _, _, rest3 = rest2.partition(b"\x00")
                        _, _, val = rest3.partition(b"\x00")
                        try:
                            if comp == 1:
                                val = zlib.decompress(val)
                            texts[kw.decode("latin-1")] = val.decode("utf-8", "replace")
                        except Exception:
                            pass
                elif ctype == b"zTXt":
                    kw, _, rest = data.partition(b"\x00")
                    if rest:
                        try:
                            texts[kw.decode("latin-1")] = zlib.decompress(rest[1:]).decode("utf-8", "replace")
                        except Exception:
                            pass
                if ctype == b"IEND":
                    break
    except Exception:
        pass
    return texts


# ---------------------------------------------------------------------------
# ComfyUI 条目解析
# ---------------------------------------------------------------------------
def _resolve_text(prompt, nid, depth=0):
    if depth > 10:
        return ""
    node = prompt.get(nid)
    if not node:
        return ""
    ct = node.get("class_type", "")
    inputs = node.get("inputs", {})
    if ct == "CLIPTextEncode":
        v = inputs.get("text", "")
        if isinstance(v, str):
            return v
        if isinstance(v, list) and len(v) == 2:
            return _resolve_text(prompt, v[0], depth + 1)
    if ct == "StringConcatenate":
        def g(x):
            if isinstance(x, str):
                return x
            if isinstance(x, list) and len(x) == 2:
                return _resolve_text(prompt, x[0], depth + 1)
            return ""
        return g(inputs.get("string_a", "")) + inputs.get("delimiter", "") + g(inputs.get("string_b", ""))
    for k in ("artist_tags", "character_tags", "clothing_tags", "background_tags", "pose_tags"):
        if k in inputs and isinstance(inputs[k], str):
            return inputs[k]
    for v in inputs.values():
        if isinstance(v, str) and len(v) > 3:
            return v
        if isinstance(v, list) and len(v) == 2:
            r = _resolve_text(prompt, v[0], depth + 1)
            if r:
                return r
    return ""


def _norm_lora(name):
    """规范化 LoRA 名：去目录路径、去扩展名，并过滤明显不是 LoRA 名的值。
    注：不把纯数字（如 "1","0"）当垃圾过滤——它们可能是合法 LoRA 名（如 Anima/画风/self 下的 1.safetensors）。
    布尔/JSON/null 结构值仍过滤。"""
    n = os.path.splitext(os.path.basename(str(name).replace("\\", "/")))[0].strip()
    if not n:
        return ""
    low = n.lower()
    # 过滤布尔/结构体/JSON 片段等非 LoRA 值（数字除外）
    if low in ("true", "false", "none", "null", "[]", "{}"):
        return ""
    if len(n) > 80:
        return ""  # 过长：被误提取的整段提示词
    if n.startswith("{") or n.startswith("["):
        return ""
    return n


# 明确的 LoRA 加载节点（精确匹配避免误把提示词当 LoRA）
_LORA_NODE_TYPES = {"loraloader", "loraloadermodelonly"}
# Weilin/Anima 的 LoRA 堆叠节点：widgets 里存 lobby 列表 JSON（含 hidden 字段标记激活状态）
_LORA_STACK_NODE_HINTS = ("weilinpromptuionlylorastack", "animamultiloraloader")


def _parse_lora_json_field(text):
    """解析 weilin 堆叠节点的 lora 列表 JSON，返回其中实际激活（hidden=false）的 lora 名列表。
    输入可能是 JSON 数组字符串，也可能本身已是 list。"""
    if not text:
        return []
    try:
        if isinstance(text, str):
            data = json.loads(text)
        else:
            data = text
    except Exception:
        return []
    names = []
    if isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue
            # hidden == true 表示堆在节点里但未激活；未隐藏（false/缺失）视为激活
            if item.get("hidden"):
                continue
            name = item.get("name") or item.get("lora") or ""
            if name:
                names.append(str(name))
    return names


def _extract_loras(prompt, workflow):
    """从 ComfyUI prompt / workflow 提取 LoRA 名列表（原始未规范化）
    - 标准 LoRA 节点：LoraLoader / LoraLoaderModelOnly，取 inputs.lora_name（字符串）
    - Weilin/Anima 堆叠节点：从 widgets 的 lora 列表 JSON 解析，仅取激活（hidden!=true）的 lora
    - workflow: 仅精确匹配节点类型，从 widgets_values 里找第一个合法字符串作 lora_name
    不做 class_type 含 'lora' 的宽松匹配（避免把 ControlNet-Lora 等非 LoraLoader 节点误判为 LoRA）。
    """
    loras = []
    if isinstance(prompt, dict):
        for node in prompt.values():
            ct = str(node.get("class_type", "")).lower()
            i = node.get("inputs", {}) or {}
            v = i.get("lora_name")
            if ct in _LORA_NODE_TYPES and isinstance(v, str):
                loras.append(v)
            elif any(h in ct for h in _LORA_STACK_NODE_HINTS):
                # 堆叠节点：从 lora_str / lora_list_json / temp_lora_str 解析激活的 lora
                for key in ("lora_list_json", "lora_str"):
                    loras.extend(_parse_lora_json_field(i.get(key)))
    if workflow:
        try:
            for n in workflow.get("nodes", []):
                t = str(n.get("type", "")).lower()
                wv = n.get("widgets_values") or []
                if t in _LORA_NODE_TYPES:
                    for item in wv:
                        if isinstance(item, str) and len(item) > 2 and not item.startswith("{") and " " not in item.strip():
                            loras.append(item)
                            break
                elif any(h in t for h in _LORA_STACK_NODE_HINTS):
                    # 堆叠节点：widgets[0]=激活列表(lora_str), widgets[1]=临时列表(temp_lora_str)
                    # 优先用第一个（当前激活）的 JSON
                    for item in wv:
                        if isinstance(item, list):
                            loras.extend(_parse_lora_json_field(item))
                            break
                    else:
                        if wv and isinstance(wv[0], str) and wv[0].startswith("["):
                            loras.extend(_parse_lora_json_field(wv[0]))
        except Exception:
            pass
    return loras


def _extract_comfyui(prompt, workflow):
    info = {"positive": "", "negative": "", "model": "", "params": {}, "loras": []}
    info["loras"] = sorted({x for x in (_norm_lora(y) for y in _extract_loras(prompt, workflow)) if x})
    if isinstance(prompt, dict):
        def sval(nid, key):
            v = prompt.get(nid, {}).get("inputs", {}).get(key, "")
            return v if isinstance(v, str) else ""
        parts = [p for p in (sval("148", "artist_tags").strip().rstrip(",").strip(),
                             sval("200", "positive").strip(),
                             sval("150", "character_tags").strip(),
                             sval("152", "clothing_tags").strip(),
                             sval("182", "background_tags").strip(),
                             sval("184", "pose_tags").strip()) if p]
        info["positive"] = ", ".join(parts)
        info["negative"] = sval("7", "text")
        if not info["positive"] or not info["negative"]:
            for node in prompt.values():
                # 兼容 KSampler / KSamplerAdvanced（视频/图片工作流都可能用）
                if node.get("class_type") in ("KSampler", "KSamplerAdvanced"):
                    pos = node.get("inputs", {}).get("positive")
                    neg = node.get("inputs", {}).get("negative")
                    if isinstance(pos, list) and len(pos) == 2 and not info["positive"]:
                        info["positive"] = _resolve_text(prompt, pos[0])
                    if isinstance(neg, list) and len(neg) == 2 and not info["negative"]:
                        info["negative"] = _resolve_text(prompt, neg[0])
                    if info["positive"] and info["negative"]:
                        break
        for node in prompt.values():
            ct = node.get("class_type", "")
            i = node.get("inputs", {})
            if ct == "UNETLoader" and i.get("unet_name"):
                info["model"] = i["unet_name"]
                break
            if ct in ("CheckpointLoaderSimple", "CheckpointLoader") and i.get("ckpt_name"):
                info["model"] = i["ckpt_name"]
                break
        for node in prompt.values():
            ct = node.get("class_type", "")
            i = node.get("inputs", {})
            if ct in ("KSampler", "KSamplerAdvanced"):
                for k in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise", "noise_seed"):
                    if k in i:
                        info["params"][k] = i[k]
                # 统一采样器键名：SD WebUI 用 "sampler"，ComfyUI 是 "sampler_name"
                if "sampler_name" in i:
                    info["params"]["sampler"] = i["sampler_name"]
            elif ct == "EmptyLatentImage":
                if "width" in i:
                    info["params"]["width"] = i["width"]
                if "height" in i:
                    info["params"]["height"] = i["height"]
    if workflow and not info["positive"]:
        try:
            nodes = {n["id"]: n for n in workflow.get("nodes", [])}
            def wv(nid, idx):
                n = nodes.get(nid)
                if n and "widgets_values" in n and idx < len(n["widgets_values"]):
                    return n["widgets_values"][idx]
                return None
            info["positive"] = str(wv(200, 0) or "")
            for n in workflow.get("nodes", []):
                t = n.get("type", "")
                if t in ("UNETLoader", "CheckpointLoaderSimple", "CheckpointLoader") and n.get("widgets_values"):
                    info["model"] = str(n["widgets_values"][0])
                    break
        except Exception:
            pass
    return info


# ---------------------------------------------------------------------------
# SD WebUI 条目解析（parameters 串）
# ---------------------------------------------------------------------------
def _parse_sd_parameters(text):
    info = {"positive": "", "negative": "", "model": "", "params": {}, "loras": []}
    if not text:
        return info
    lines = text.split("\n")
    # 定位参数行（包含 Steps: 且包含 Sampler 的最后一行），它不属于提示词
    param_idx = None
    for i, l in enumerate(lines):
        if "Steps:" in l and "Sampler" in l:
            param_idx = i
    body = lines[:param_idx] if param_idx is not None else lines
    # 从提示词正文里分离 positive / negative
    neg_idx = None
    for i, l in enumerate(body):
        if l.startswith("Negative prompt:"):
            neg_idx = i
            break
    if neg_idx is None:
        pos_lines = body
    else:
        pos_lines = body[:neg_idx]
        neg = body[neg_idx][len("Negative prompt:"):].strip()
        for l in body[neg_idx + 1:]:
            if l.strip():
                neg += " " + l.strip()
        info["negative"] = neg.strip()
    info["positive"] = "\n".join(x for x in pos_lines if x.strip()).strip()
    # 参数行解析（Steps/Sampler/CFG/Seed/Size/Model...）
    if param_idx is not None:
        kv = {}
        for part in lines[param_idx].split(","):
            if ":" in part:
                k, v = part.split(":", 1)
                kv[k.strip().lower()] = v.strip()
        info["model"] = kv.get("model", "")
        info["params"] = {
            "seed": kv.get("seed"), "steps": kv.get("steps"),
            "sampler": kv.get("sampler"), "cfg": kv.get("cfg scale"),
            "width": None, "height": None,
        }
        m = re.search(r"(\d+)\s*[xX×]\s*(\d+)", kv.get("size", ""))
        if m:
            info["params"]["width"] = int(m.group(1))
            info["params"]["height"] = int(m.group(2))
    info["loras"] = sorted({x for x in (_norm_lora(y) for y in re.findall(r"<lora:([^:>]+)", info["positive"])) if x})
    return info


# ---------------------------------------------------------------------------
# 索引构建（增量）
# ---------------------------------------------------------------------------
def _index_file(source):
    return os.path.join(INDEX_DIR, source + "_index.json")


def _load_cached_index(source):
    try:
        with open(_index_file(source), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        pass
    # 容错恢复：JSON 损坏（如写入中断）时，用 raw_decode 逐个解出完整条目，损坏点停止
    fp = _index_file(source)
    if not os.path.exists(fp):
        return None
    try:
        with open(fp, encoding="utf-8", errors="replace") as f:
            txt = f.read()
    except Exception:
        return None
    try:
        import json as _j
        dec = _j.JSONDecoder()
        txt2 = txt.strip()
        if not txt2.startswith("["):
            return None
        txt2 = txt2[1:]  # 跳过 '['
        out = []
        idx = 0
        n = len(txt2)
        while idx < n:
            # 跳过空白/逗号
            while idx < n and txt2[idx] in " \t\r\n,":
                idx += 1
            if idx >= n:
                break
            if txt2[idx] == "]":
                break
            try:
                obj, end = dec.raw_decode(txt2, idx)
            except Exception:
                break  # 损坏点，停止
            if isinstance(obj, dict):
                out.append(obj)
            idx = end
            # 安全上限
            if len(out) > 200000:
                break
        if out:
            # 自愈：用原子写把恢复出的完好数据写回，替换损坏文件
            try:
                _save_index(source, out)
            except Exception:
                pass
        return out if out else None
    except Exception:
        return None


def _save_index(source, entries):
    os.makedirs(INDEX_DIR, exist_ok=True)
    fp = _index_file(source)
    tmp = fp + ".tmp"
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, fp)  # 原子替换：磁盘上永远只有完整文件
    except Exception:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass
        raise


def _iter_files(root, exts, skip=("thumbnails",), recursive=True):
    """遍历目录下指定扩展名的文件（不进入 skip 目录）"""
    if not root or not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() not in skip]
        for fn in filenames:
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)


def _iter_image_files(root):
    return _iter_files(root, IMAGE_EXTS)


def _iter_video_files(root):
    return _iter_files(root, VIDEO_EXTS)


def _scan_directory(source, root, parser, exts=IMAGE_EXTS):
    """source: comfyui|webui|video；parser(path)->entry 增量更新；exts 决定扫描的文件扩展名"""
    cached = _load_cached_index(source) or []
    cache_map = {}
    for e in cached:
        cache_map[(e["file"], e.get("dir", ""))] = e
    files = list(_iter_files(root, exts))
    total = len(files)
    with _indexes_lock:
        _indexes[source] = {"entries": [], "ready": False, "total": total, "scanned": 0}
    new_entries = []
    for i, fp in enumerate(files):
        try:
            st = os.stat(fp)
        except OSError:
            continue
        rel = os.path.relpath(fp, root).replace("\\", "/")
        key = (rel, os.path.dirname(rel))
        cached_e = cache_map.get(key)
        if cached_e and cached_e.get("_k") == [st.st_mtime_ns, st.st_size] and "loras" in cached_e:
            new_entries.append(cached_e)
        else:
            entry = parser(fp)
            entry.update({"file": rel, "mtime": st.st_mtime, "size": st.st_size,
                          "_k": [st.st_mtime_ns, st.st_size]})
            new_entries.append(entry)
        if i % 200 == 0:
            with _indexes_lock:
                _indexes[source]["scanned"] = i
    with _indexes_lock:
        _indexes[source] = {"entries": new_entries, "ready": True, "total": total, "scanned": total}
    _save_index(source, new_entries)
    return new_entries


def _parser_comfyui(fp):
    texts = read_png_texts(fp)
    info = {"positive": "", "negative": "", "model": "", "params": {}}
    prompt = workflow = None
    try:
        if texts.get("prompt"):
            prompt = json.loads(texts["prompt"])
    except Exception:
        pass
    try:
        if texts.get("workflow"):
            workflow = json.loads(texts["workflow"])
    except Exception:
        pass
    info.update(_extract_comfyui(prompt, workflow))
    if not info["positive"]:
        info["positive"] = texts.get("Description", "")[:500]
    return info


def _parser_webui(fp):
    texts = read_png_texts(fp)
    return _parse_sd_parameters(texts.get("parameters") or texts.get("Description") or "")


def _parser_video(fp):
    """视频条目：关联同名 png 提取元数据（如 output/video/ComfyUI_00001_.mp4 ↔ output/ComfyUI_00001_.png）。
    视频文件本身不含 workflow/prompt，部分元数据来自同名 png 的提示词/模型/参数。"""
    info = {"positive": "", "negative": "", "model": "", "params": {}, "type": "video"}
    base = os.path.splitext(fp)[0]
    # 在同目录及上一级目录找同名 png（ComfyUI 视频通常在 output/video/ 子目录，png 在 output/ 根）
    dirs = [os.path.dirname(fp)]
    parent = os.path.dirname(os.path.dirname(fp))
    if parent != os.path.dirname(fp):
        dirs.append(parent)
    candidates = []
    for d in dirs:
        for cand_base in (base, base.rstrip("_")):
            for ext in (".png", ".jpg", ".jpeg", ".webp"):
                candidates.append(cand_base + ext)
                candidates.append(os.path.join(d, os.path.basename(cand_base) + ext))
    seen = set()
    for cand in candidates:
        if cand in seen:
            continue
        seen.add(cand)
        if os.path.isfile(cand):
            try:
                png_info = _parser_comfyui(cand)
                if png_info.get("positive") or png_info.get("model") or png_info.get("params"):
                    info.update(png_info)
                    info["type"] = "video"
                    break
            except Exception:
                pass
    return info


def start_scan(source, force=False, refresh=False):
    """惰性启动扫描（幂等）：
    内存已就绪 -> 直接返回；磁盘有缓存 -> 加载进内存返回（不启动线程）；
    两者都没有 -> 启动后台线程扫描。force=True 时删除缓存全量重建。
    refresh=True 时即使内存/缓存已就绪，也启动后台增量扫描
    （_scan_directory 复用缓存中未变化文件，仅解析新文件，速度快）。"""
    if source == "comfyui":
        root = folder_paths.get_output_directory() if folder_paths else None
        parser = _parser_comfyui
        exts = IMAGE_EXTS
    elif source == "video":
        # 视频：优先 ComfyUI output/video 子目录（主 root）
        root = None
        if folder_paths and folder_paths.get_output_directory():
            vdir = os.path.join(folder_paths.get_output_directory(), "video")
            if os.path.isdir(vdir):
                root = vdir
            else:
                root = folder_paths.get_output_directory()
        if not root:
            root = webui_outputs_dir()
        parser = _parser_video
        exts = VIDEO_EXTS
    else:
        root = webui_outputs_dir()
        parser = _parser_webui
        exts = IMAGE_EXTS
    if not root:
        return {"ok": False, "error": "来源目录不可用"}
    if force:
        fp = _index_file(source)
        if os.path.exists(fp):
            try:
                os.remove(fp)
            except Exception:
                pass
    with _indexes_lock:
        st = _indexes.get(source)
        if st and st.get("ready") and not force and not refresh:
            return {"ok": True, "ready": True}
    if not force and not refresh:
        cached = _load_cached_index(source)
        if cached is not None:
            with _indexes_lock:
                _indexes[source] = {"entries": cached, "ready": True,
                                    "total": len(cached), "scanned": len(cached)}
            return {"ok": True, "ready": True}
    threading.Thread(target=_scan_directory, args=(source, root, parser, exts), daemon=True).start()
    return {"ok": True, "ready": False}


def get_index(source):
    with _indexes_lock:
        st = _indexes.get(source)
        if st and st.get("ready"):
            return st["entries"], True
    cached = _load_cached_index(source)
    if cached is not None:
        with _indexes_lock:
            _indexes[source] = {"entries": cached, "ready": True,
                                "total": len(cached), "scanned": len(cached)}
        return cached, True
    return [], False


def scan_status(source):
    with _indexes_lock:
        st = _indexes.get(source)
    if not st:
        return {"source": source, "ready": False, "scanned": 0, "total": 0}
    return {"source": source, "ready": st["ready"], "scanned": st["scanned"], "total": st["total"]}


# ---------------------------------------------------------------------------
# 查询
# ---------------------------------------------------------------------------
def _fav_set():
    global _favs
    if _favs is None:
        try:
            with open(os.path.join(USERDATA, "favorites.json"), encoding="utf-8") as f:
                _favs = set(tuple(x) for x in json.load(f))
        except Exception:
            _favs = set()
    return _favs


def save_fav(source, file, on):
    """普通星标 = 收藏到「默认收藏」文件夹"""
    key = {"source": source, "file": file}
    data = load_folders()
    data.setdefault("默认收藏", [])
    if on:
        if key not in data["默认收藏"]:
            data["默认收藏"].append(key)
    else:
        data["默认收藏"] = [k for k in data["默认收藏"] if k != key]
    save_folders(data)
    return {"ok": True}


def query(source, q="", model="", sampler="", lora="", steps_min=None, steps_max=None,
          cfg_min=None, cfg_max=None, min_w=None, min_h=None, fav_only=False,
          sort="newest", page=1, page_size=60, tags=""):
    entries, ready = get_index(source)
    if q:
        q = q.lower()
        entries = [e for e in entries
                   if q in str(e.get("file") or "").lower()
                   or q in str(e.get("positive") or "").lower()
                   or q in str(e.get("negative") or "").lower()]
    if tags:
        # 多 tag 组合（AND）：条目必须包含全部 tag
        tag_list = [t.strip().lower() for t in tags.split(",") if t.strip()]
        entries = [e for e in entries
                   if all(t in str(e.get("positive") or "").lower() for t in tag_list)]
    if model:
        entries = [e for e in entries if str(e.get("model") or "") == model]
    if sampler:
        entries = [e for e in entries
                   if str(e.get("params", {}).get("sampler")
                          or e.get("params", {}).get("sampler_name") or "") == sampler]
    if lora:
        entries = [e for e in entries if lora in (e.get("loras") or [])]
    if steps_min is not None or steps_max is not None:
        entries = [e for e in entries
                   if (steps_min is None or _num(e, "steps", steps_min) >= steps_min)
                   and (steps_max is None or _num(e, "steps", steps_max) <= steps_max)]
    if cfg_min is not None or cfg_max is not None:
        entries = [e for e in entries
                   if (cfg_min is None or _num(e, "cfg", cfg_min) >= cfg_min)
                   and (cfg_max is None or _num(e, "cfg", cfg_max) <= cfg_max)]
    if min_w:
        entries = [e for e in entries if _num(e, "width", min_w) >= min_w]
    if min_h:
        entries = [e for e in entries if _num(e, "height", min_h) >= min_h]
    if fav_only:
        favs = {(it.get("source", "comfyui"), it.get("file"))
                for it in load_folders().get("默认收藏", [])}
        entries = [e for e in entries if (source, e.get("file")) in favs]
    if sort == "newest":
        entries.sort(key=lambda e: e.get("mtime", 0), reverse=True)
    elif sort == "oldest":
        entries.sort(key=lambda e: e.get("mtime", 0))
    elif sort == "name":
        entries.sort(key=lambda e: str(e.get("file", "")).lower())
    elif sort == "random":
        import random
        entries = random.sample(entries, len(entries)) if entries else entries
    total = len(entries)
    start = (int(page) - 1) * int(page_size)
    items = [{k: e.get(k) for k in ("file", "mtime", "size", "positive", "negative", "model", "params", "loras")}
             for e in entries[start:start + int(page_size)]]
    return {"total": total, "page": int(page), "items": items, "ready": ready}


def _num(e, key, default=0):
    try:
        v = e.get("params", {}).get(key)
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def models(source):
    entries, _ = get_index(source)
    counter = {}
    for e in entries:
        m = str(e.get("model") or "").strip()
        if m:
            counter[m] = counter.get(m, 0) + 1
    return [{"model": m, "count": n} for m, n in sorted(counter.items(), key=lambda kv: -kv[1])]


def samplers(source):
    entries, _ = get_index(source)
    counter = {}
    for e in entries:
        s = str(e.get("params", {}).get("sampler")
                or e.get("params", {}).get("sampler_name") or "").strip()
        if s:
            counter[s] = counter.get(s, 0) + 1
    return [{"sampler": s, "count": n} for s, n in sorted(counter.items(), key=lambda kv: -kv[1])]


def loras(source):
    """LoRA 列表：合并【图片实际用过】与【磁盘上存在】两类 LoRA。
    - 图片用过：count = 使用次数
    - 磁盘存在但未用过：count = 0（新加入的 LoRA 也能在此列表看到）
    - 磁盘已不存在的（被拷贝/删除）：不列出
    - folder：该 LoRA 相对于 loras 根目录的文件夹（如 'Anima/画风'，根目录为 ''），供前端按文件夹分组
    按安装时间（文件 mtime）倒序，同时间按使用次数。"""
    entries, _ = get_index(source)
    counter = {}
    for e in entries:
        for l in (e.get("loras") or []):
            l = str(l).strip()
            if l:
                counter[l] = counter.get(l, 0) + 1
    dmap = _lora_disk_map(source)
    # 磁盘上所有存在的 LoRA 名集合（含未使用过的）
    all_disk = set(dmap.keys())
    items = []
    seen = set()
    for l, n in counter.items():
        if l not in all_disk:
            continue  # 磁盘已不存在该 LoRA：从列表清除
        info = dmap[l]
        items.append((l, n, info["mtime"], info["folder"]))
        seen.add(l)
    # 补齐：磁盘存在但图片从未用过的（count=0）
    for l in all_disk:
        if l not in seen:
            info = dmap[l]
            items.append((l, 0, info["mtime"], info["folder"]))
    items.sort(key=lambda x: (-x[2], -x[1]))  # 按安装时间倒序（新装的在前），同时间按使用次数
    return [{"lora": l, "count": n, "mtime": mt, "folder": folder} for l, n, mt, folder in items]


_LORA_EXTS = (".safetensors", ".pt", ".ckpt", ".bin", ".pth", ".sft")


def _lora_disk_map(source):
    """扫描磁盘 LoRA 目录：{规范化名: {mtime, folder}}；目录不可用时返回空 dict（表示不过滤）。
    folder = 相对 loras 根的目录（'Anima/画风'，根目录为 ''），供前端按文件夹分组。
    ComfyUI 源只用 ComfyUI 自己的 loras 根目录（排除 extra_model_paths 中挂载的 webui 等额外路径）。"""
    roots = _lora_roots(source)
    m = {}
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        try:
            for dirpath, _dirnames, filenames in os.walk(root):
                rel_dir = os.path.relpath(dirpath, root).replace("\\", "/")
                if rel_dir == ".":
                    rel_dir = ""
                for fn in filenames:
                    if fn.lower().endswith(_LORA_EXTS):
                        base = os.path.splitext(fn)[0].strip()
                        if base:
                            try:
                                m[base] = {"mtime": os.path.getmtime(os.path.join(dirpath, fn)),
                                           "folder": rel_dir}
                            except Exception:
                                pass
        except Exception:
            continue
    return m


def _lora_roots(source):
    """返回应扫描的 LoRA 根目录列表。
    - comfyui：仅 ComfyUI 自己的 models/loras（不扫 extra_model_paths 挂载进来的 webui 路径）
    - webui：SD WebUI 的 models/Lora"""
    if source == "comfyui":
        if folder_paths is not None:
            try:
                mdir = folder_paths.models_dir
                if mdir:
                    return [os.path.join(mdir, "loras")]
            except Exception:
                pass
        return []
    else:
        root = os.path.join(load_settings().get("webui_root", ""), "models", "Lora")
        return [root] if os.path.isdir(root) else []


def stats():
    out = {"comfyui": {"total": 0, "models": []}, "webui": {"total": 0, "models": []}}
    for src in ("comfyui", "webui"):
        entries, ready = get_index(src)
        out[src]["total"] = len(entries) if ready else 0
        out[src]["ready"] = ready
        out[src]["models"] = models(src)[:12]
    return out


def image_path(source, file):
    if source == "comfyui":
        root = folder_paths.get_output_directory() if folder_paths else None
    elif source == "video":
        root = None
        if folder_paths and folder_paths.get_output_directory():
            vdir = os.path.join(folder_paths.get_output_directory(), "video")
            root = vdir if os.path.isdir(vdir) else folder_paths.get_output_directory()
        if not root:
            root = webui_outputs_dir()
    else:
        root = webui_outputs_dir()
    if not root:
        return None
    safe = os.path.normpath(file)
    fp = os.path.join(root, safe)
    if os.path.isfile(fp) and os.path.commonpath([root, fp]) == os.path.normpath(root):
        return fp
    return None


def raw_metadata(source, file):
    fp = image_path(source, file)
    if not fp:
        return ""
    texts = read_png_texts(fp)
    return "\n\n".join("%s:\n%s" % (k, v) for k, v in texts.items())


# ---------------------------------------------------------------------------
# tag 统计（从历史图片提示词统计，按使用次数降序）
# ---------------------------------------------------------------------------
_tag_cache = {}


def load_blacklist():
    try:
        with open(os.path.join(USERDATA, "blacklist.json"), encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def save_blacklist(items):
    os.makedirs(USERDATA, exist_ok=True)
    with open(os.path.join(USERDATA, "blacklist.json"), "w", encoding="utf-8") as f:
        json.dump([str(x).strip() for x in items if str(x).strip()], f, ensure_ascii=False, indent=1)
    return {"ok": True}


def tag_stats(source, search="", limit=200):
    """历史图片 tag 统计（排除黑名单），按使用次数降序"""
    now = time.time()
    cached = _tag_cache.get(source)
    if not cached or now - cached[0] > 120:
        entries, _ = get_index(source)
        counter = {}
        blacklist = set(load_blacklist())
        for e in entries:
            pos = str(e.get("positive") or "")
            for tag in re.split(r"[,，\n]", pos):
                tag = tag.strip().lower()
                if not tag or len(tag) > 80 or tag in blacklist:
                    continue
                counter[tag] = counter.get(tag, 0) + 1
        _tag_cache[source] = (now, counter)
    counter = _tag_cache[source][1]
    items = sorted(counter.items(), key=lambda kv: -kv[1])
    s = (search or "").strip().lower()
    if s:
        items = [x for x in items if s in x[0]]
    return [{"tag": t, "count": c} for t, c in items[:limit]]


# ---------------------------------------------------------------------------
# 画师（Artist）：从提示词提取 @tag，聚合统计 + 封面/权重持久化
# ---------------------------------------------------------------------------
_ARTIST_RE = re.compile(r"@([A-Za-z0-9_\-\. ]+)")
_artist_cache = {}
ARTIST_COVER_FILE = os.path.join(USERDATA, "artist_covers.json")
ARTIST_WEIGHT_FILE = os.path.join(USERDATA, "artist_weights.json")


def _norm_artist(tag):
    """规范化画师 tag：去 @、去首尾空格、压缩多个空格、转小写作为键"""
    t = str(tag or "").strip()
    if t.startswith("@"):
        t = t[1:]
    t = re.sub(r"\s+", " ", t).strip()
    if not t or t.lower() in ("none", "null"):
        return ""
    return t


def _artist_cover_key(tag):
    return tag.lower().replace(" ", "_")


def load_artist_covers():
    try:
        with open(ARTIST_COVER_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_artist_cover(tag, file):
    d = load_artist_covers()
    d[_artist_cover_key(tag)] = file
    os.makedirs(USERDATA, exist_ok=True)
    with open(ARTIST_COVER_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return {"ok": True}


def load_artist_weights():
    try:
        with open(ARTIST_WEIGHT_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_artist_weight(tag, weight):
    d = load_artist_weights()
    d[_artist_cover_key(tag)] = weight
    os.makedirs(USERDATA, exist_ok=True)
    with open(ARTIST_WEIGHT_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return {"ok": True}


def _artist_entries(source):
    """聚合画师：name(显示名) / key / count / last_use / first_use / cover（默认最新）"""
    now = time.time()
    cached = _artist_cache.get(source)
    if cached and now - cached[0] < 120:
        return cached[1]
    entries, _ = get_index(source)
    agg = {}
    for e in entries:
        pos = str(e.get("positive") or "")
        mtimes = e.get("mtime") or 0
        for m in _ARTIST_RE.finditer(pos):
            name = _norm_artist(m.group(1))
            if not name:
                continue
            key = name.lower()
            a = agg.get(key)
            fname = str(e.get("file") or "")
            if a is None:
                a = agg[key] = {"name": name, "key": key, "count": 0,
                                "last_use": mtimes, "first_use": mtimes, "cover": fname}
            a["count"] += 1
            if mtimes > a["last_use"]:
                a["last_use"] = mtimes
            if not a["first_use"] or mtimes < a["first_use"]:
                a["first_use"] = mtimes
            # 默认封面 = 最新一张（同 count 时保留首见）
            if not a["cover"] or mtimes >= a.get("_cover_mt", 0):
                a["cover"] = fname
                a["_cover_mt"] = mtimes
    # 应用持久化封面
    covers = load_artist_covers()
    for k, a in agg.items():
        if k in covers and covers[k]:
            a["cover"] = covers[k]
    _artist_cache[source] = (time.time(), list(agg.values()))
    return list(agg.values())


# 画师库（WeiLin 导出 md）：@画师名 + 中文备注。ComfyUI 没用过的画师也从这里占位。
ARTIST_LIB_FILE = os.environ.get("GALLERY4_WEILIN_FILE", "") or _data_file("WeiLin画师库-用户自定义.md")
ARTIST_NOTES_FILE = os.path.join(USERDATA, "artist_notes.json")

# Danbooru 收录数轻量映射（画师名 → post_count，供画师模块显示；独立小文件不触发画师库全量加载）
ARTLIB_POSTS_FILE = _data_file("danbooru_posts.json")
_db_posts_cache = None


def db_post_counts():
    """画师名(小写) → Danbooru 收录作品数（模块级缓存，只读一次）。"""
    global _db_posts_cache
    if _db_posts_cache is None:
        try:
            with open(ARTLIB_POSTS_FILE, encoding="utf-8") as f:
                _db_posts_cache = json.load(f)
        except Exception:
            _db_posts_cache = {}
    return _db_posts_cache


def _load_artist_lib():
    """解析 WeiLin 画师库 md：{key: {"name": 无@名, "display": @名, "note": 中文备注}}"""
    out = {}
    try:
        with open(ARTIST_LIB_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                # 行格式：| 序号 | `@画师名` | 备注 |
                if not line.startswith("|") or "`@`" in line or "| --- |" in line:
                    continue
                # 跳到 Anima 写法列（第二个反引号字段）
                if line.count("`") < 2:
                    continue
                try:
                    display = line.split("`")[1].strip()  # @画师名
                    note = line.split("|")[-2].strip() if line.count("|") >= 4 else ""
                except Exception:
                    continue
                if not display.startswith("@"):
                    continue
                name = display[1:].strip()
                key = name.lower()
                if not key or key in ("none", "null"):
                    continue
                out[key] = {"name": name, "display": display, "note": note or "", "from_lib": True}
    except Exception:
        pass
    return out


def load_artist_notes():
    try:
        with open(ARTIST_NOTES_FILE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_artist_note(tag, note):
    d = load_artist_notes()
    d[_artist_cover_key(tag)] = str(note or "").strip()
    os.makedirs(USERDATA, exist_ok=True)
    with open(ARTIST_NOTES_FILE, "w", encoding="utf-8") as f:
        json.dump(d, f, ensure_ascii=False, indent=1)
    return {"ok": True}


ARTIST_FAVS_FILE = os.path.join(USERDATA, "artist_favs.json")


def load_artist_favs():
    try:
        with open(ARTIST_FAVS_FILE, encoding="utf-8") as f:
            return set(json.load(f))
    except Exception:
        return set()


def save_artist_fav(tag, on):
    d = load_artist_favs()
    key = _artist_cover_key(tag)
    if on:
        d.add(key)
    else:
        d.discard(key)
    os.makedirs(USERDATA, exist_ok=True)
    with open(ARTIST_FAVS_FILE, "w", encoding="utf-8") as f:
        json.dump(sorted(d), f, ensure_ascii=False, indent=1)
    return {"ok": True, "fav": on}


def artists(source, sort="count", search="", page=1, page_size=0):
    """画师列表：ComfyUI 用过的 + WeiLin 画师库占位（未用过的 count=0）。
    返回 {"items": [...], "total": N}；page_size<=0 时不切片返回全部。
    字段：name/display(@名)/note(备注,用户覆盖优先)/count/cover/weight/used"""
    items = list(_artist_entries(source))
    used_keys = {a["key"] for a in items}
    wmap = load_artist_weights()
    lib = _load_artist_lib()
    notes = load_artist_notes()
    lib_map = {a["key"]: a for a in items}
    # 补入 WeiLin 库中未用过的画师（占位 count=0）
    for key, li in lib.items():
        if key in lib_map:
            continue
        items.append({"name": li["name"], "key": key, "count": 0,
                      "last_use": 0, "first_use": 0, "cover": "",
                      "weight": wmap.get(key, 1.0), "used": False})
    for a in items:
        a["weight"] = wmap.get(a["key"], 1.0)
        li = lib.get(a["key"])
        a["used"] = a["key"] in used_keys
        # 备注：用户自定义覆盖 > WeiLin 库备注 > 空
        note = notes.get(a["key"])
        if note is None or note == "":
            note = li["note"] if li else ""
        a["note"] = note or ""
        a["display"] = (li["display"] if li else "@" + a["name"])
    favs = load_artist_favs()
    for a in items:
        a["fav"] = a["key"] in favs
    dbp = db_post_counts()
    for a in items:
        a["db_posts"] = dbp.get(a["key"], 0) or 0
    s = (search or "").strip().lower()
    if s:
        items = [a for a in items if s in a["key"] or s in a["name"].lower() or s in str(a["note"]).lower()]
    if sort == "count":
        items.sort(key=lambda x: -x["count"])
    elif sort == "last":
        items.sort(key=lambda x: -(x["last_use"] or 0))
    elif sort == "first":
        items.sort(key=lambda x: x["first_use"] or 0)
    elif sort == "random":
        # 固定种子：翻页/刷新时顺序稳定，但每次会话不同
        rng = random.Random(20260206)
        rng.shuffle(items)
    else:
        items.sort(key=lambda x: x["name"].lower())
    total = len(items)
    if page_size and page_size > 0:
        page = max(1, page)
        start = (page - 1) * page_size
        items = items[start:start + page_size]
    return {"items": items, "total": total}


def artist_top_tags(source, tag, max_show=400, limit=40):
    """该画师所有作品的正向提示词 tag 统计（按使用次数降序）。
    - 屏蔽总体使用次数 > max_show 的通用高频 tag（如 masterpiece/1girl 等）
    - 返回 [{tag, count}]，最多 limit 条；tag 为 '全部 tag' 时统计整个源。"""
    entries, _ = get_index(source)
    atag = ("@" + tag).lower() if not tag.startswith("@") else tag.lower()
    counter = {}
    for e in entries:
        pos = str(e.get("positive") or "")
        if atag not in pos.lower():
            continue
        for t in re.split(r"[,，\n]", pos):
            t = t.strip()
            if not t or len(t) > 60:
                continue
            t_low = t.lower()
            if t_low.startswith("@"):      # 排除画师 tag 本身
                continue
            counter[t] = counter.get(t, 0) + 1
        if len(counter) > 30000:
            break
    # 屏蔽总体 > max_show 的通用 tag（简单方式：全源统计一遍高频 tag）
    if max_show and max_show > 0:
        # 共现过滤：> max_show 意味着该 tag 在全部图里出现太多，属通用词
        global_cnt = {}
        gsum = 0
        for e in entries:
            for t in re.split(r"[,，\n]", str(e.get("positive") or "")):
                t = t.strip()
                if t and len(t) <= 60 and not t.lower().startswith("@"):
                    global_cnt[t.lower()] = global_cnt.get(t.lower(), 0) + 1
            gsum += 1
            if gsum > 6000:
                break
        items = [{"tag": t, "count": c} for t, c in counter.items()
                 if global_cnt.get(t.lower(), 0) <= max_show]
    else:
        items = [{"tag": t, "count": c} for t, c in counter.items()]
    items.sort(key=lambda x: -x["count"])
    return items[:limit]


def load_folders():
    try:
        with open(os.path.join(USERDATA, "folders.json"), encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):      # 旧格式迁移
            data = {"默认收藏": data}
            save_folders(data)
        return data if isinstance(data, dict) else {"默认收藏": []}
    except Exception:
        return {"默认收藏": []}


def save_folders(data):
    os.makedirs(USERDATA, exist_ok=True)
    with open(os.path.join(USERDATA, "folders.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def folder_list():
    return {k: len(v) for k, v in load_folders().items()}


def folder_create(name):
    name = (name or "").strip()
    if not name:
        return {"ok": False, "error": "名称不能为空"}
    data = load_folders()
    if name in data:
        return {"ok": False, "error": "收藏夹已存在"}
    data[name] = []
    save_folders(data)
    return {"ok": True, "folders": folder_list()}


def folder_delete(name):
    data = load_folders()
    if name in data:
        del data[name]
        save_folders(data)
    return {"ok": True, "folders": folder_list()}


def folder_rename(old, new):
    new = (new or "").strip()
    data = load_folders()
    if old in data and new and new not in data:
        data[new] = data.pop(old)
        save_folders(data)
    return {"ok": True, "folders": folder_list()}


def folder_add(name, items):
    """items: [{source, file}, ...]"""
    data = load_folders()
    data.setdefault(name, [])
    for it in items or []:
        key = {"source": str(it.get("source", "comfyui")), "file": str(it.get("file", ""))}
        if key["file"] and key not in data[name]:
            data[name].append(key)
    save_folders(data)
    return {"ok": True, "count": len(data[name]), "folders": folder_list()}


def folder_remove(name, items):
    data = load_folders()
    if name in data:
        keys = [{"source": str(it.get("source", "comfyui")), "file": str(it.get("file", ""))}
                for it in items or []]
        data[name] = [k for k in data[name] if k not in keys]
        save_folders(data)
    return {"ok": True, "folders": folder_list()}


def folder_images(name, source=""):
    """收藏夹条目（关联索引取完整信息，按收藏顺序）"""
    data = load_folders()
    items = data.get(name, [])
    by_source = {}
    for src in ("comfyui", "webui"):
        entries, ready = get_index(src)
        if ready:
            by_source[src] = {e.get("file"): e for e in entries}
    out = []
    for it in items:
        src = it.get("source", "comfyui")
        e = by_source.get(src, {}).get(it.get("file"))
        if e:
            entry = {k: e.get(k) for k in ("file", "mtime", "size", "positive", "negative", "model", "params")}
            entry["source"] = src
            out.append(entry)
    if source:
        out = [e for e in out if e.get("source") == source]
    return out


def folder_contains(source, file):
    """返回包含该图片的收藏夹名列表"""
    data = load_folders()
    return [name for name, items in data.items()
            if {"source": source, "file": file} in items]


# ---------------------------------------------------------------------------
# 文件系统浏览（设置里选择目录用，只读）
# ---------------------------------------------------------------------------
def list_drives():
    import string
    return [d + ":\\" for d in string.ascii_uppercase if os.path.exists(d + ":\\")]


def list_dir(path):
    path = os.path.normpath(path or "")
    if not os.path.isdir(path):
        return {"error": "目录不存在: %s" % path}
    try:
        dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])
    except PermissionError:
        return {"error": "无权限访问"}
    parent = os.path.dirname(path)
    if parent == path:
        parent = None
    return {
        "path": path,
        "parent": parent,
        "dirs": [{"name": d,
                  "has_outputs": os.path.isdir(os.path.join(path, d, "outputs")),
                  "has_models": os.path.isdir(os.path.join(path, d, "models"))}
                 for d in dirs],
    }


# ---------------------------------------------------------------------------
# Danbooru 画师库（懒加载：点开模块才读数据，不点击不加载）
# ---------------------------------------------------------------------------
ARTLIB_JSONL = _data_file("artists.jsonl")
ARTLIB_SAMPLE_CSV = _data_file("sample_artists_classified.csv")
ARTLIB_TOPPOST_CSV = _data_file("toppost_artists_classified.csv")
ARTLIB_SPECIALTY_FILE = _data_file("artlib_specialty.jsonl")
_artlib = {"loaded": False, "artists": [], "by_id": {}, "cls": {}, "spec": {}}  # 懒加载缓存

# 分类字段白名单（CSV 列）
ARTLIB_CLS_FIELDS = ("archetype", "origin", "main_subject", "main_theme",
                     "main_technique", "main_composition", "rating_label")


def artlib_facets():
    """加载画师库并返回各分类的可选值 + 总数。首次调用才读磁盘（懒加载）。"""
    _artlib_ensure()
    facets = {}
    for f in ARTLIB_CLS_FIELDS:
        vals = {}
        for c in _artlib["cls"].values():
            v = (c.get(f) or "").strip()
            if v:
                vals[v] = vals.get(v, 0) + 1
        facets[f] = [{"value": v, "count": n} for v, n in sorted(vals.items(), key=lambda x: -x[1])]
    return {
        "loaded": True,
        "total": len(_artlib["artists"]),
        "classified": len(_artlib["cls"]),
        "facets": facets,
    }


def artlib_search(q="", archetype="", origin="", subject="", theme="", technique="",
                  composition="", rating="", min_posts=0, sort="posts", page=1, page_size=60):
    """画师库筛选分页。q=名字搜索（子串/前缀，不区分大小写）。"""
    _artlib_ensure()
    items = []
    for a in _artlib["artists"]:
        c = _artlib["cls"].get(a["id"]) or {}
        if archetype and c.get("archetype", "") != archetype:
            continue
        if origin and c.get("origin", "") != origin:
            continue
        if subject and c.get("main_subject", "") != subject:
            continue
        if theme and c.get("main_theme", "") != theme:
            continue
        if technique and c.get("main_technique", "") != technique:
            continue
        if composition and c.get("main_composition", "") != composition:
            continue
        if rating and c.get("rating_label", "") != rating:
            continue
        if min_posts and (a.get("post_count") or 0) < min_posts:
            continue
        if q:
            ql = q.lower()
            name = a.get("name") or ""
            if ql not in name.lower():
                continue
        items.append(a)
    if sort == "name":
        items.sort(key=lambda x: (x.get("name") or "").lower())
    elif sort == "posts_asc":
        items.sort(key=lambda x: x.get("post_count") or 0)
    elif sort == "random":
        # 固定种子：翻页时顺序稳定（同一筛选结果每次随机序一致）
        rng = random.Random(20260206)
        rng.shuffle(items)
    else:  # posts（默认作品数降序）
        items.sort(key=lambda x: -(x.get("post_count") or 0))
    total = len(items)
    page = max(1, page)
    page_size = min(max(1, page_size or 60), 300)
    start = (page - 1) * page_size
    page_items = items[start:start + page_size]
    out = []
    for a in page_items:
        c = _artlib["cls"].get(a["id"]) or {}
        sp = _artlib["spec"].get(a["id"]) or {}
        out.append({
            "id": a["id"], "name": a["name"], "posts": a.get("post_count") or 0,
            "deprecated": bool(a.get("is_deprecated")),
            "profile": c.get("profile_line") or "",
            "rating": c.get("rating_label") or sp.get("ero_label") or "",
            "specialty": sp.get("specialty") or c.get("profile_line") or "",
            "sex_tags": sp.get("sex_tags") or [],
            "soft_tags": sp.get("soft_tags") or [],
            "ero_ratio": sp.get("ero_ratio") or 0,
            "archetype": c.get("archetype") or "",
            "origin": c.get("origin") or "",
            "subject": c.get("main_subject") or "",
            "theme": c.get("main_theme") or "",
            "technique": c.get("main_technique") or "",
            "composition": c.get("main_composition") or "",
            "crypto": c.get("top_copyrights") or "",
            "sig": c.get("signature_tags") or "",
            "url": c.get("url") or "",
        })
    return {"items": out, "total": total}


def artlib_preview(source="comfyui", names=()):
    """画师库本地预览：返回 {画师名(小写): 本地索引中该 tag 的最近一张图 file}。
    复用 _artist_entries（含 2 分钟缓存与持久化封面）。"""
    want = {str(n).strip().lower() for n in names if n and str(n).strip()}
    if not want:
        return {}
    out = {}
    try:
        for a in _artist_entries(source):
            if a["key"] in want and a.get("cover"):
                out[a["key"]] = a["cover"]
    except Exception:
        pass
    return out


def _artlib_ensure():
    """懒加载：读 artists.jsonl + 两个分类 CSV（各字段只取非空首个值）。"""
    if _artlib["loaded"]:
        return
    artists = []
    try:
        with open(ARTLIB_JSONL, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                    artists.append({"id": o.get("id"), "name": o.get("name") or "",
                                    "post_count": o.get("post_count") or 0,
                                    "is_deprecated": bool(o.get("is_deprecated"))})
                except Exception:
                    pass
    except Exception:
        pass
    cls = {}
    # 用 csv 模块解析分类 CSV（字段内含逗号；utf-8-sig 去掉 BOM）
    try:
        import csv as _csv
        for csv in (ARTLIB_SAMPLE_CSV, ARTLIB_TOPPOST_CSV):
            with open(csv, encoding="utf-8-sig", newline="") as f:
                rd = _csv.DictReader(f)
                for row in rd:
                    try:
                        aid = int(row.get("id") or 0)
                        if not aid:
                            continue
                        info = cls.get(aid) or {}
                        for k in ARTLIB_CLS_FIELDS:
                            v = (row.get(k) or "").strip()
                            if v and not info.get(k):
                                info[k] = v
                        for k in ("profile_line", "top_copyrights", "signature_tags", "rating_label", "url"):
                            v = (row.get(k) or "").strip()
                            if v and not info.get(k):
                                info[k] = v
                        cls[aid] = info
                    except Exception:
                        pass
    except Exception:
        pass
    # by_id 索引
    by_id = {}
    for a in artists:
        by_id[a["id"]] = a
    # 涩涩特色定性（本地数据分析生成）
    spec = {}
    try:
        with open(ARTLIB_SPECIALTY_FILE, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    o = json.loads(line)
                except Exception:
                    continue
                aid = o.get("id")
                if not aid:
                    continue
                spec[aid] = {
                    "specialty": o.get("specialty") or "",
                    "sex_tags": o.get("sex_tags") or [],
                    "soft_tags": o.get("soft_tags") or [],
                    "ero_label": o.get("ero_label") or "",
                    "ero_ratio": o.get("ero_ratio") or 0,
                }
    except Exception:
        pass
    _artlib.update({"loaded": True, "artists": artists, "by_id": by_id, "cls": cls, "spec": spec})
