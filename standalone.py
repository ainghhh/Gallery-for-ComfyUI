# -*- coding: utf-8 -*-
"""
Gallery4ComfyUI · 独立启动器
=============================
不启动 ComfyUI，单独运行图库服务（自带小型 aiohttp 服务器）。

用法：
    python standalone.py [--port 8288] [--output <ComfyUI output 目录>]

- 默认端口 8288（避免与 ComfyUI 的 8188 冲突）
- 默认 output 目录 = 绘世整合包 ComfyUI 的 output（可用 --output 覆盖）
- 启动后浏览器访问 http://127.0.0.1:8288/
"""
import os, sys, json, subprocess, zipfile, datetime, asyncio, time, tempfile, re
from types import SimpleNamespace

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PLUGIN_DIR)

import gallery_core as G

# ---------------------------------------------------------------------------
# 参数解析
# ---------------------------------------------------------------------------
PORT = 8288
COMFY_OUTPUT = None
argv = sys.argv[1:]
i = 0
while i < len(argv):
    if argv[i] == "--port" and i + 1 < len(argv):
        PORT = int(argv[i + 1]); i += 2; continue
    if argv[i] == "--output" and i + 1 < len(argv):
        COMFY_OUTPUT = argv[i + 1]; i += 2; continue
    i += 1

# ---------------------------------------------------------------------------
# ComfyUI output 目录（folder_paths 替身）
# ---------------------------------------------------------------------------
_CANDIDATES = [
    os.environ.get("GALLERY_COMFY_OUTPUT", ""),
    r"D:\ai\Stable Diffusion\ComfyUI\ComfyUI-aki-v3\ComfyUI\output",
]
if COMFY_OUTPUT:
    _CANDIDATES.insert(0, COMFY_OUTPUT)
COMFY_OUTPUT = next((p for p in _CANDIDATES if p and os.path.isdir(p)), "")
if not COMFY_OUTPUT:
    print("[!] 未找到 ComfyUI output 目录，请用 --output 指定：")
    print("    python standalone.py --output D:\\path\\to\\ComfyUI\\output")
    sys.exit(1)
G.folder_paths = SimpleNamespace(
    get_output_directory=lambda: COMFY_OUTPUT,
    models_dir=r"D:\ai\Stable Diffusion\ComfyUI\ComfyUI-aki-v3\ComfyUI\models",
)
print("[*] ComfyUI output 目录:", COMFY_OUTPUT)

# ---------------------------------------------------------------------------
# 路由（与 ComfyUI 插件内完全一致）
# ---------------------------------------------------------------------------
from aiohttp import web

P = "/gallery4comfyui"
WEB_DIR = os.path.join(PLUGIN_DIR, "web")


def _int(v, default=None):
    try:
        return int(v) if v not in (None, "") else default
    except Exception:
        return default


def _float(v, default=None):
    try:
        return float(v) if v not in (None, "") else default
    except Exception:
        return default


def _json(resp, code=200):
    return web.json_response(resp, status=code, dumps=lambda o: json.dumps(o, ensure_ascii=False))


def _safe_name(name):
    n = (name or "").replace("\\", "/").lstrip("/")
    parts = [p for p in n.split("/") if p not in ("", ".", "..") and not p.endswith(":")]
    return "/".join(parts)


def _pick_native_folder():
    """用 Win32 SHBrowseForFolder 弹出原生“选择文件夹”对话框（可新建文件夹），返回路径或空串。
    不经过 PowerShell（其 stdout 管道在模态对话框下会挂起），直接用 ctypes 调用，可靠。"""
    try:
        import ctypes
        from ctypes import POINTER, byref
        class BROWSEINFOW(ctypes.Structure):
            _fields_ = [("hwndOwner", ctypes.c_void_p), ("pidlRoot", ctypes.c_void_p),
                        ("pszDisplayName", ctypes.c_wchar_p), ("lpszTitle", ctypes.c_wchar_p),
                        ("ulFlags", ctypes.c_uint), ("lpfn", ctypes.c_void_p), ("lParam", ctypes.c_void_p),
                        ("iImage", ctypes.c_int)]
        ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # STA
        shell32 = ctypes.windll.shell32
        shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
        shell32.SHBrowseForFolderW.argtypes = [POINTER(BROWSEINFOW)]
        shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        bi = BROWSEINFOW()
        bi.hwndOwner = None
        bi.lpszTitle = "选择打包保存文件夹（可新建）"
        bi.ulFlags = 0x40  # BIF_NEWDIALOGSTYLE
        pidl = shell32.SHBrowseForFolderW(byref(bi))
        if pidl:
            buf = ctypes.create_unicode_buffer(260)
            if shell32.SHGetPathFromIDListW(pidl, buf):
                return buf.value.strip()
        return ""
    except Exception:
        return ""


app = web.Application()


async def page_index(request):
    fp = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(fp):
        resp = web.FileResponse(fp)
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return web.Response(status=404, text="index.html not found")


async def api_settings(request):
    s = G.load_settings()
    return _json({"webui_root": s.get("webui_root", ""),
                  "configured": bool(s.get("webui_root"))})


async def api_settings_save(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    r = G.save_settings(data.get("webui_root", ""))
    return _json(r, 200 if r.get("ok") else 400)


async def api_scan(request):
    source = request.query.get("source", "comfyui")
    force = request.query.get("force", "0") == "1"
    refresh = request.query.get("refresh", "0") == "1"
    r = G.start_scan(source, force=force, refresh=refresh)
    return _json(r)


async def api_status(request):
    return _json({
        "comfyui": G.scan_status("comfyui"),
        "webui": G.scan_status("webui"),
        "settings": G.load_settings(),
    })


async def api_gallery(request):
    q = request.query
    source = q.get("source", "comfyui")
    r = G.query(
        source,
        q=q.get("q", ""),
        model=q.get("model", ""),
        sampler=q.get("sampler", ""),
        lora=q.get("lora", ""),
        steps_min=_int(q.get("steps_min")),
        steps_max=_int(q.get("steps_max")),
        cfg_min=_float(q.get("cfg_min")),
        cfg_max=_float(q.get("cfg_max")),
        min_w=_int(q.get("min_w")),
        min_h=_int(q.get("min_h")),
        fav_only=q.get("fav", "0") == "1",
        sort=q.get("sort", "newest"),
        page=_int(q.get("page"), 1),
        page_size=min(_int(q.get("page_size"), 60) or 60, 1000),
        tags=q.get("tags", ""),
    )
    return _json(r)


async def api_models(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.models(source)})


async def api_samplers(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.samplers(source)})


async def api_loras(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.loras(source)})


async def api_stats(request):
    return _json(G.stats())


async def api_fav_list(request):
    return _json({"items": [{"source": a, "file": b} for a, b in G._fav_set()]})


async def api_fav_toggle(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    r = G.save_fav(data.get("source", "comfyui"), _safe_name(data.get("file", "")), bool(data.get("on", True)))
    return _json(r)


async def api_image(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    fp = G.image_path(source, file)
    if not fp:
        return _json({"error": "not found"}, 404)
    return web.FileResponse(fp)


async def api_open_folder(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    fp = G.image_path(source, file)
    if not fp:
        return _json({"error": "not found"}, 404)
    try:
        subprocess.Popen(["explorer", "/select,", fp])
        return _json({"ok": True})
    except Exception as e:
        return _json({"ok": False, "error": str(e)})


async def api_meta(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    return _json({"metadata": G.raw_metadata(source, file)})


async def api_boot(request):
    results = {}
    for src in ("comfyui", "webui"):
        results[src] = G.start_scan(src, refresh=True)
    return _json({"ok": True, "booted": results})


async def api_tags(request):
    source = request.query.get("source", "comfyui")
    search = request.query.get("search", "")
    limit = min(_int(request.query.get("limit"), 200) or 200, 1000)
    return _json({"items": G.tag_stats(source, search=search, limit=limit)})


async def api_blacklist_get(request):
    return _json({"items": G.load_blacklist()})


async def api_blacklist_save(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return _json(G.save_blacklist(data.get("items", [])))


async def api_folders_get(request):
    return _json({"folders": G.folder_list()})


async def api_folders_images(request):
    name = request.query.get("folder", "")
    source = request.query.get("source", "")
    return _json({"items": G.folder_images(name, source)})


async def api_folders_of(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    return _json({"folders": G.folder_contains(source, file)})


async def api_folders_manage(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    action = data.get("action")
    if action == "create":
        return _json(G.folder_create(data.get("name", "")))
    if action == "delete":
        return _json(G.folder_delete(data.get("name", "")))
    if action == "rename":
        return _json(G.folder_rename(data.get("old", ""), data.get("new", "")))
    if action == "add":
        return _json(G.folder_add(data.get("folder", ""), data.get("items", [])))
    if action == "remove":
        return _json(G.folder_remove(data.get("folder", ""), data.get("items", [])))
    return _json({"ok": False, "error": "未知操作"}, 400)


async def api_fs_drives(request):
    return _json({"drives": G.list_drives()})


async def api_fs_list(request):
    path = request.query.get("path", "")
    r = G.list_dir(path)
    return _json(r, 200 if "error" not in r else 400)


async def api_fs_mkdir(request):
    """在当前浏览目录下新建文件夹（打包保存用）"""
    try:
        data = await request.json()
    except Exception:
        data = {}
    parent = (data.get("path") or "").strip()
    name = (data.get("name") or "").strip()
    if not parent or not name:
        return _json({"error": "缺少路径或名称"}, 400)
    if not os.path.isdir(parent):
        return _json({"error": "父目录不存在"}, 400)
    name = name.replace("\\", "/").replace("/", "")
    if not name:
        return _json({"error": "名称无效"}, 400)
    target = os.path.join(parent, name)
    try:
        os.makedirs(target, exist_ok=True)
        return _json({"ok": True, "path": target})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


async def api_pick_dir(request):
    """弹出 Windows 原生“选择文件夹”对话框（可新建文件夹），返回选中路径"""
    path = await asyncio.to_thread(_pick_native_folder)
    if path and os.path.isdir(path):
        return _json({"path": path})
    return _json({"path": ""})


async def api_zip(request):
    """打包图片到用户选择的目标文件夹。
    body: {target, items} 选中打包 或 {target, query} 筛选结果整体打包；by_model=true 按大模型分子目录"""
    try:
        data = await request.json()
    except Exception:
        return _json({"error": "bad json"}, 400)
    target = (data.get("target") or "").strip()
    if not target or not os.path.isdir(target):
        return _json({"error": "目标目录不存在"}, 400)
    by_model = bool(data.get("by_model", False))
    entries = []  # (磁盘路径, 子目录)
    n_candidates = 0
    if data.get("items"):
        for it in data["items"]:
            fp = G.image_path(it.get("source", "comfyui"), _safe_name(it.get("file", "")))
            if fp:
                entries.append((fp, ""))
    elif data.get("query"):
        q = data["query"]
        r = G.query(
            q.get("source", "comfyui"),
            q=q.get("q", ""), model=q.get("model", ""), sampler=q.get("sampler", ""),
            lora=q.get("lora", ""),
            steps_min=_int(q.get("steps_min")), steps_max=_int(q.get("steps_max")),
            cfg_min=_float(q.get("cfg_min")), cfg_max=_float(q.get("cfg_max")),
            min_w=_int(q.get("min_w")), min_h=_int(q.get("min_h")),
            fav_only=q.get("fav", "0") == "1", tags=q.get("tags", ""),
            sort="newest", page=1, page_size=10000,
        )
        n_candidates = len(r.get("items", []))
        for it in r.get("items", []):
            fp = G.image_path(q.get("source", "comfyui"), _safe_name(it.get("file", "")))
            if fp:
                sub = str(it.get("model") or "未分类") if by_model else ""
                entries.append((fp, sub))
    else:
        return _json({"error": "no items"}, 400)
    if not entries:
        msg = "没有可打包的图片"
        if n_candidates:
            msg += "（候选 %d 个，磁盘均找不到文件）" % n_candidates
        return _json({"error": msg}, 400)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zpath = os.path.join(target, "gallery_export_%s.zip" % ts)
    try:
        with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
            seen = set()
            for fp, sub in entries:
                name = os.path.basename(fp)
                if name in seen:
                    name = "%d_%s" % (len(seen), name)
                seen.add(name)
                if sub:
                    safe = re.sub(r'[\\/:*?"<>|]', "_", sub).strip() or "未分类"
                    z.write(fp, os.path.join(safe, name))
                else:
                    z.write(fp, name)
        return _json({"ok": True, "zip": zpath, "count": len(entries)})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
app.router.add_get(P + "/api/fs/list", api_fs_list)
app.router.add_post(P + "/api/fs/mkdir", api_fs_mkdir)
app.router.add_post(P + "/api/fs/pick-dir", api_pick_dir)
app.router.add_post(P + "/api/zip", api_zip)
app.router.add_get(P + "/api/fs/drives", api_fs_drives)
app.router.add_post(P + "/api/folders", api_folders_manage)
app.router.add_get(P + "/api/folders/of", api_folders_of)
app.router.add_get(P + "/api/folders/images", api_folders_images)
app.router.add_get(P + "/api/folders", api_folders_get)
app.router.add_post(P + "/api/blacklist", api_blacklist_save)
app.router.add_get(P + "/api/blacklist", api_blacklist_get)
app.router.add_get(P + "/api/tags", api_tags)
app.router.add_get(P + "/api/boot", api_boot)
app.router.add_get(P + "/api/meta", api_meta)
app.router.add_get(P + "/api/image/open-folder", api_open_folder)
app.router.add_get(P + "/api/image", api_image)
app.router.add_post(P + "/api/fav", api_fav_toggle)
app.router.add_get(P + "/api/fav", api_fav_list)
app.router.add_get(P + "/api/stats", api_stats)
app.router.add_get(P + "/api/loras", api_loras)
app.router.add_get(P + "/api/samplers", api_samplers)
app.router.add_get(P + "/api/models", api_models)
app.router.add_get(P + "/api/gallery", api_gallery)
app.router.add_get(P + "/api/status", api_status)
app.router.add_get(P + "/api/scan", api_scan)
app.router.add_post(P + "/api/settings", api_settings_save)
app.router.add_get(P + "/api/settings", api_settings)
app.router.add_get(P + "/", page_index)


if __name__ == "__main__":
    print("[*] Gallery4ComfyUI 独立服务启动中...")
    print("[*] 打开浏览器: http://127.0.0.1:%d%s/" % (PORT, P))
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
