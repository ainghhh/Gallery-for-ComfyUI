# -*- coding: utf-8 -*-
"""
Gallery4ComfyUI —— ComfyUI 自定义节点插件（发布版）
==============================================
- ComfyUI 画布添加「Gallery4ComfyUI」节点 → 点「打开图库」按钮 → 新窗口打开图库
- 图库为双来源标签页：ComfyUI 图片 / SD WebUI 图片（首次运行需配置 SD WebUI 根目录）
- 页面与 API 由插件在 ComfyUI 进程内提供（无需独立服务）
- 支持搜索、按模型/采样器/步数/CFG/尺寸筛选、多排序、收藏、统计

安装：将本目录放入 ComfyUI/custom_nodes/Gallery4ComfyUI 后重启 ComfyUI。
"""
import os, json, threading, subprocess, zipfile, datetime, asyncio, time, tempfile, re

from server import PromptServer
from aiohttp import web

from . import gallery_core as G

WEB_DIRECTORY = "./js"

# ---------------------------------------------------------------------------
# 节点
# ---------------------------------------------------------------------------
class Gallery4ComfyUIOpener:
    """图库打开器：前端 JS 提供「打开图库」按钮"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "webui_url": ("STRING", {"default": ""}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "noop"
    OUTPUT_NODE = True
    CATEGORY = "Gallery"

    def noop(self, webui_url):
        return ()


NODE_CLASS_MAPPINGS = {"Gallery4ComfyUIOpener": Gallery4ComfyUIOpener}
NODE_DISPLAY_NAME_MAPPINGS = {"Gallery4ComfyUIOpener": "Gallery4ComfyUI"}
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
P = "/gallery4comfyui"
WEB_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web")


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
    """安全化相对路径：去掉绝对路径/盘符/.. 前缀，保留子目录（如 txt2img-images/xx.png）"""
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
        ctypes.windll.ole32.CoInitializeEx(None, 0x2)  # STA（对话框中需要）
        shell32 = ctypes.windll.shell32
        shell32.SHBrowseForFolderW.restype = ctypes.c_void_p
        shell32.SHBrowseForFolderW.argtypes = [POINTER(BROWSEINFOW)]
        shell32.SHGetPathFromIDListW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p]
        bi = BROWSEINFOW()
        bi.hwndOwner = None
        bi.lpszTitle = "选择打包保存文件夹（可新建）"
        bi.ulFlags = 0x40  # BIF_NEWDIALOGSTYLE：使用新式对话框（含“新建文件夹”按钮）
        pidl = shell32.SHBrowseForFolderW(byref(bi))
        if pidl:
            buf = ctypes.create_unicode_buffer(260)
            if shell32.SHGetPathFromIDListW(pidl, buf):
                return buf.value.strip()
        return ""
    except Exception:
        return ""


@PromptServer.instance.routes.get(P + "/")
async def page_index(request):
    fp = os.path.join(WEB_DIR, "index.html")
    if os.path.isfile(fp):
        resp = web.FileResponse(fp)
        resp.headers["Cache-Control"] = "no-store"
        return resp
    return web.Response(status=404, text="index.html not found")


@PromptServer.instance.routes.get(P + "/api/settings")
async def api_settings(request):
    s = G.load_settings()
    return _json({"webui_root": s.get("webui_root", ""),
                  "configured": bool(s.get("webui_root"))})


@PromptServer.instance.routes.post(P + "/api/settings")
async def api_settings_save(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    r = G.save_settings(data.get("webui_root", ""))
    return _json(r, 200 if r.get("ok") else 400)


@PromptServer.instance.routes.get(P + "/api/scan")
async def api_scan(request):
    source = request.query.get("source", "comfyui")
    force = request.query.get("force", "0") == "1"
    refresh = request.query.get("refresh", "0") == "1"
    r = G.start_scan(source, force=force, refresh=refresh)
    return _json(r)


@PromptServer.instance.routes.get(P + "/api/status")
async def api_status(request):
    return _json({
        "comfyui": G.scan_status("comfyui"),
        "webui": G.scan_status("webui"),
        "settings": G.load_settings(),
    })


@PromptServer.instance.routes.get(P + "/api/gallery")
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


@PromptServer.instance.routes.get(P + "/api/models")
async def api_models(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.models(source)})


@PromptServer.instance.routes.get(P + "/api/samplers")
async def api_samplers(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.samplers(source)})


@PromptServer.instance.routes.get(P + "/api/loras")
async def api_loras(request):
    source = request.query.get("source", "comfyui")
    return _json({"items": G.loras(source)})


@PromptServer.instance.routes.get(P + "/api/stats")
async def api_stats(request):
    return _json(G.stats())


@PromptServer.instance.routes.get(P + "/api/fav")
async def api_fav_list(request):
    return _json({"items": [{"source": a, "file": b} for a, b in G._fav_set()]})


@PromptServer.instance.routes.post(P + "/api/fav")
async def api_fav_toggle(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    r = G.save_fav(data.get("source", "comfyui"), _safe_name(data.get("file", "")), bool(data.get("on", True)))
    return _json(r)


@PromptServer.instance.routes.get(P + "/api/image")
async def api_image(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    fp = G.image_path(source, file)
    if not fp:
        return _json({"error": "not found"}, 404)
    return web.FileResponse(fp)


@PromptServer.instance.routes.get(P + "/api/image/open-folder")
async def api_open_folder(request):
    """在文件管理器中打开图片所在文件夹并选中该文件（Windows）"""
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


@PromptServer.instance.routes.get(P + "/api/meta")
async def api_meta(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    return _json({"metadata": G.raw_metadata(source, file)})


@PromptServer.instance.routes.get(P + "/api/boot")
async def api_boot(request):
    """点节点按钮时调用：惰性启动图库；每次打开都触发后台增量扫描（新图片立即可见）"""
    results = {}
    for src in ("comfyui", "webui"):
        results[src] = G.start_scan(src, refresh=True)
    return _json({"ok": True, "booted": results})


@PromptServer.instance.routes.get(P + "/api/tags")
async def api_tags(request):
    source = request.query.get("source", "comfyui")
    search = request.query.get("search", "")
    limit = min(_int(request.query.get("limit"), 200) or 200, 1000)
    return _json({"items": G.tag_stats(source, search=search, limit=limit)})


@PromptServer.instance.routes.get(P + "/api/blacklist")
async def api_blacklist_get(request):
    return _json({"items": G.load_blacklist()})


@PromptServer.instance.routes.post(P + "/api/blacklist")
async def api_blacklist_save(request):
    try:
        data = await request.json()
    except Exception:
        data = {}
    return _json(G.save_blacklist(data.get("items", [])))


@PromptServer.instance.routes.get(P + "/api/folders")
async def api_folders_get(request):
    return _json({"folders": G.folder_list()})


@PromptServer.instance.routes.get(P + "/api/folders/images")
async def api_folders_images(request):
    name = request.query.get("folder", "")
    source = request.query.get("source", "")
    return _json({"items": G.folder_images(name, source)})


@PromptServer.instance.routes.get(P + "/api/folders/of")
async def api_folders_of(request):
    source = request.query.get("source", "comfyui")
    file = _safe_name(request.query.get("file", ""))
    return _json({"folders": G.folder_contains(source, file)})


@PromptServer.instance.routes.post(P + "/api/folders")
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


@PromptServer.instance.routes.get(P + "/api/fs/drives")
async def api_fs_drives(request):
    return _json({"drives": G.list_drives()})


@PromptServer.instance.routes.get(P + "/api/fs/list")
async def api_fs_list(request):
    path = request.query.get("path", "")
    r = G.list_dir(path)
    return _json(r, 200 if "error" not in r else 400)


@PromptServer.instance.routes.post(P + "/api/fs/mkdir")
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


@PromptServer.instance.routes.get(P + "/api/fs/open-dir")
async def api_fs_open_dir(request):
    """在资源管理器中打开一个目录"""
    path = request.query.get("path", "").strip()
    if not path or not os.path.isdir(path):
        return _json({"error": "目录不存在"}, 400)
    try:
        subprocess.Popen(["explorer", path])
        return _json({"ok": True})
    except Exception as e:
        return _json({"ok": False, "error": str(e)}, 500)


@PromptServer.instance.routes.post(P + "/api/fs/pick-dir")
async def api_pick_dir(request):
    """弹出 Windows 原生“选择文件夹”对话框（可新建文件夹），返回选中路径"""
    path = await asyncio.to_thread(_pick_native_folder)
    if path and os.path.isdir(path):
        return _json({"path": path})
    return _json({"path": ""})


@PromptServer.instance.routes.post(P + "/api/zip")
async def api_zip(request):
    """打包图片到用户选择的目标文件夹（打包在后台线程执行，不阻塞服务）。
    body: {target, items} 选中打包 或 {target, query} 筛选结果整体打包；
    by_model=true 按大模型分子目录；name=指定 zip 文件名（默认时间戳）。"""
    try:
        data = await request.json()
    except Exception:
        return _json({"error": "bad json"}, 400)
    resp = await asyncio.to_thread(_pack_images, data)
    code = resp.pop("_code", 200)
    return _json(resp, code)


def _pack_images(data):
    """同步执行：收集文件 + 写 zip（在后台线程运行）。返回 (resp_dict, code)"""
    target = (data.get("target") or "").strip()
    if not target or not os.path.isdir(target):
        return {"error": "目标目录不存在", "_code": 400}
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
        return {"error": "no items", "_code": 400}
    if not entries:
        msg = "没有可打包的图片"
        if n_candidates:
            msg += "（候选 %d 个，磁盘均找不到文件）" % n_candidates
        return {"error": msg, "_code": 400}
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    zname = (data.get("name") or "").strip()
    if zname:
        safe_n = re.sub(r'[\\/:*?"<>|]', "_", zname).strip() or "gallery_export"
        zpath = os.path.join(target, safe_n + ".zip")
        if os.path.exists(zpath):
            zpath = os.path.join(target, "%s_%s.zip" % (safe_n, ts))
    else:
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
        return {"ok": True, "zip": zpath, "count": len(entries), "_code": 200}
    except Exception as e:
        return {"ok": False, "error": str(e), "_code": 500}


# 注意：不在 ComfyUI 启动时做任何扫描 —— 图库为懒启动，
# 仅在用户点节点按钮（/api/boot）或打开图库页面时才触发索引。
