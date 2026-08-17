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
import os, json, time, zlib, threading, re

try:
    import folder_paths
except Exception:
    folder_paths = None

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
USERDATA = os.path.join(PLUGIN_DIR, "userdata")
SETTINGS_FILE = os.path.join(USERDATA, "settings.json")
INDEX_DIR = os.path.join(USERDATA, "index")

_indexes = {}          # source -> {entries: [...], ready: bool, total: int, scanned: int}
_indexes_lock = threading.Lock()
_favs = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp")


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


def _extract_comfyui(prompt, workflow):
    info = {"positive": "", "negative": "", "model": "", "params": {}}
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
                if node.get("class_type") == "KSampler":
                    pos = node.get("inputs", {}).get("positive")
                    neg = node.get("inputs", {}).get("negative")
                    if isinstance(pos, list) and len(pos) == 2 and not info["positive"]:
                        info["positive"] = _resolve_text(prompt, pos[0])
                    if isinstance(neg, list) and len(neg) == 2 and not info["negative"]:
                        info["negative"] = _resolve_text(prompt, neg[0])
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
            if ct == "KSampler":
                for k in ("seed", "steps", "cfg", "sampler_name", "scheduler", "denoise"):
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
    info = {"positive": "", "negative": "", "model": "", "params": {}}
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
        return None


def _save_index(source, entries):
    os.makedirs(INDEX_DIR, exist_ok=True)
    with open(_index_file(source), "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)


def _iter_image_files(root):
    """遍历目录下所有图片（不进入 thumbnails 目录）"""
    if not root or not os.path.isdir(root):
        return
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d.lower() != "thumbnails"]
        for fn in filenames:
            if fn.lower().endswith(IMAGE_EXTS):
                yield os.path.join(dirpath, fn)


def _scan_directory(source, root, parser):
    """source: comfyui|webui；parser(path)->entry 增量更新"""
    cached = _load_cached_index(source) or []
    cache_map = {}
    for e in cached:
        cache_map[(e["file"], e.get("dir", ""))] = e
    files = list(_iter_image_files(root))
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
        if cached_e and cached_e.get("_k") == [st.st_mtime_ns, st.st_size]:
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


def start_scan(source, force=False):
    """惰性启动扫描（幂等）：
    内存已就绪 -> 直接返回；磁盘有缓存 -> 加载进内存返回（不启动线程）；
    两者都没有 -> 启动后台线程扫描。force=True 时删除缓存全量重建。"""
    if source == "comfyui":
        root = folder_paths.get_output_directory() if folder_paths else None
        parser = _parser_comfyui
    else:
        root = webui_outputs_dir()
        parser = _parser_webui
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
        if st and st.get("ready") and not force:
            return {"ok": True, "ready": True}
    if not force:
        cached = _load_cached_index(source)
        if cached is not None:
            with _indexes_lock:
                _indexes[source] = {"entries": cached, "ready": True,
                                    "total": len(cached), "scanned": len(cached)}
            return {"ok": True, "ready": True}
    threading.Thread(target=_scan_directory, args=(source, root, parser), daemon=True).start()
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


def query(source, q="", model="", sampler="", steps_min=None, steps_max=None,
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
    items = [{k: e.get(k) for k in ("file", "mtime", "size", "positive", "negative", "model", "params")}
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
# 收藏夹（多文件夹，支持拖拽/批量）
# ---------------------------------------------------------------------------
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
