# sample_data（极小示例，仅用于预览 UI）

这里只有 **20 位画师**，用来让别人装完插件就能看到「画师库」标签页长什么样。

**不是完整数据**，也不含任何用户私有内容——完整数据请自己生成：

```bash
python tools/build_artist_db.py
```

## 用法

插件默认读插件目录下的 `data/`。想用这份示例：

```bash
# 方式一：指环境变量（推荐，不动插件目录）
set GALLERY4_DATA_DIR=<插件目录>/sample_data      # Windows
export GALLERY4_DATA_DIR=<插件目录>/sample_data   # Linux / macOS
```

或直接把 `sample_data/*` 复制进 `data/`。

## 数据来源

画师名与作品数抓取自 Danbooru 公开 API（`/tags.json`，category=1 即画师）。
`post_count` 为抓取时刻的快照。
