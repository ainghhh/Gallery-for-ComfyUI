# -*- coding: utf-8 -*-
"""
自建画师库数据（可选）。

仓库**不附带** Danbooru 画师数据（约 17 MB，且属于抓取内容），需要的话自己跑一次：

    python tools/build_artist_db.py                 # 默认输出到 ./data/
    python tools/build_artist_db.py --min-posts 100 --out D:\\mygallerydata

产物：<out>/artists.jsonl   —— 每行一个画师 {id, name, post_count, ...}
插件读取顺序：环境变量 GALLERY4_DATA_DIR > 插件目录 data/

关于 Danbooru API 的两个坑（本脚本已规避）：
  1) /artists.json 的 search[post_count_gte] 会被**静默忽略**，且该接口不返回 post_count；
     所以改成 /tags.json?search[category]=1&search[order]=count 按作品数降序翻页，
     一旦某页跌破 min-posts 就停止。
  2) 匿名限流是**持续性配额**（不是并发限制）：短时突发测不出上限，
     长跑 500+ 次后必然 429。本脚本用全局令牌桶 + 上限记忆，稳定在 ~3 请求/秒。
请自行控制频率、遵守对方的使用条款。
"""
import argparse
import json
import os
import sys
import threading
import time

try:
    import requests
except ImportError:
    print("需要 requests：pip install requests")
    sys.exit(1)

BASE = "https://danbooru.donmai.us"
LIMIT = 1000
HEADERS = {"User-Agent": "Gallery4ComfyUI-artist-db-builder/1.0"}


class RateLimiter:
    """全局令牌桶：命中 429 就乘性退避并记住该速率不安全，恢复时最多爬回上限的 80%。"""

    def __init__(self, rate):
        self.lock = threading.Lock()
        self.base = 1.0 / rate
        self.interval = self.base
        self.next_t = time.time()
        self.ceiling = None

    def acquire(self):
        with self.lock:
            now = time.time()
            wait = max(0.0, self.next_t - now)
            self.next_t = max(now, self.next_t) + self.interval
        if wait > 0:
            time.sleep(wait)

    def slow(self):
        with self.lock:
            self.ceiling = 1.0 / self.interval if self.ceiling is None \
                else min(self.ceiling, 1.0 / self.interval)
            self.interval = min(self.interval * 1.6, 1.0)

    def recover(self):
        with self.lock:
            target = self.base
            if self.ceiling is not None:
                target = max(self.base, 0.8 / self.ceiling)
            if self.interval > target:
                self.interval = max(target, self.interval * 0.75)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-posts", type=int, default=50, help="只收录作品数 >= 该值的画师")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "data"), help="输出目录")
    ap.add_argument("--rate", type=float, default=3.0, help="目标请求/秒")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    dest = os.path.join(args.out, "artists.jsonl")
    sess = requests.Session()
    sess.headers.update(HEADERS)
    lim = RateLimiter(args.rate)

    rows, page, t0 = [], 1, time.time()
    while True:
        params = {"search[category]": 1, "search[order]": "count",
                  "limit": LIMIT, "page": page}
        data = None
        for attempt in range(6):
            lim.acquire()
            try:
                r = sess.get(BASE + "/tags.json", params=params, timeout=60)
            except requests.RequestException as e:
                print("  网络错误 %s，重试…" % e.__class__.__name__)
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 200:
                lim.recover()
                data = r.json()
                break
            if r.status_code in (429, 502, 503, 504, 520, 521, 522):
                lim.slow()
                time.sleep(min(2.0 * (attempt + 1), 30))
                continue
            print("HTTP %s: %s" % (r.status_code, r.text[:200]))
            return 1
        if data is None:
            print("多次重试仍失败，中止。已写入 %d 条。" % len(rows))
            break
        if not data:
            break

        stop = False
        for t in data:
            pc = t.get("post_count", 0)
            if pc < args.min_posts:
                stop = True
                continue
            rows.append({"id": t["id"], "name": t["name"], "post_count": pc,
                         "category": t.get("category"),
                         "is_deprecated": t.get("is_deprecated", False),
                         "created_at": t.get("created_at"),
                         "updated_at": t.get("updated_at")})
        print("page %-4d  rank %7d-%-7d  pc %6d..%-6d  已收 %6d  %.0fs"
              % (page, (page - 1) * LIMIT + 1, (page - 1) * LIMIT + len(data),
                 data[0]["post_count"], data[-1]["post_count"], len(rows), time.time() - t0))
        page += 1
        if stop or len(data) < LIMIT:
            break

    with open(dest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("\n完成：%d 位画师（>= %d 作品）-> %s" % (len(rows), args.min_posts, dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
