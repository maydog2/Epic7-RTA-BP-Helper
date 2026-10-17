# Windows 增量补丁

Windows 版从 `1.0.0` 开始可以接收本地补丁。用户不用每次下载约 1 GB 的完整目录；只有变化的 EXE、数据、模型或前端文件会进入补丁。Android 不使用这套流程。

打包入口有三个文件：

- `build_exe.ps1`：写入版本号，构建前端和运行时数据，再调用 PyInstaller。成品在 `dist\E7_BP_Helper\`，这个目录不进 Git。
- `launcher.py`：打进 EXE 的启动文件。它先检查 EXE 旁边的补丁，需要更新时退出；否则再启动 `backend/launcher.py`。
- `E7_BP_Helper.spec`：PyInstaller 清单，列出要放进 `_internal` 的数据、模型和前端。缺 `data/runtime` 或 `frontend/dist` 时，spec 会直接停下来。

## 一次性基线

旧版 EXE 没有补丁检测代码，不能直接吃补丁。先完整打包并发布一次：

```powershell
.\build_exe.ps1 -Version 1.0.0
```

把整个 `dist\E7_BP_Helper\` 发给用户，并另外保存一份完全相同的目录，例如 `dist\baselines\1.0.0\`。以后的补丁都相对这份已发布目录生成。

## 生成补丁

准备好新的数据、模型或前端后，打包下一个版本，再比较两份完整目录：

```powershell
.\build_exe.ps1 -Version 1.0.1
.\build_windows_patch.ps1 -Baseline dist\baselines\1.0.0 -Target dist\E7_BP_Helper
```

输出示例：

```text
dist\patches\E7_BP_Helper_patch_1.0.0_to_1.0.1.zip
dist\patches\E7_BP_Helper_patch_1.0.0_to_1.0.1.summary.json
```

摘要包含原目录大小、补丁大小，以及新增、修改、删除的文件。日志、端口文件、旧补丁、备份和 `.rar` 不会进入补丁。TensorFlow 运行库没有变化时也不会进入补丁，所以常规数据更新通常只有几十 MB。

如果 EXE 名称、`_internal` 目录或补丁协议发生变化，构建器不会生成 ZIP，并返回“需要完整更新”。这时要重新发布整个目录。

## 用户安装

1. 关闭正在运行的 E7 BP Helper。
2. 把 `E7_BP_Helper_patch_*.zip` 放到 `E7_BP_Helper.exe` 同一目录。
3. 重新打开程序。

程序会在加载模型前检查补丁。版本、协议和每个文件的 SHA-256 都匹配时，它启动独立更新任务并退出；更新任务等待旧进程结束后替换文件，然后自动重新打开程序。成功的 ZIP 会移动到 `patches_applied\`。

可以一次放多个连续补丁。每次启动只安装和当前版本匹配的一个，重启后再安装下一个。

1.0.0、1.0.1、1.0.2 的程序只认识这种精确版本补丁。它们不能读取下面的文件清单更新。升到 1.0.3 需要一次性桥接包：

```powershell
.\.venv\Scripts\python.exe workflow_scripts\build_windows_patch.py `
  --baseline dist\baselines\1.0.0 `
  --baseline dist\baselines\1.0.1 `
  --baseline dist\baselines\1.0.2 `
  --target dist\E7_BP_Helper `
  --output dist\patches\E7_BP_Helper_bridge_to_1.0.3.zip
```

用户把桥接包里的所有文件解压到 `E7_BP_Helper.exe` 同级，然后双击 `安装补丁.bat`。脚本只保留与当前版本匹配的补丁并启动程序。没有补丁功能的更早安装包仍需下载完整文件夹。

## 文件清单更新（1.0.3 起）

1.0.3 的程序会在启动时比较本机文件哈希和更新包里的完整文件清单。同一个 `E7_BP_Helper_update_*.zip` 可以覆盖所有已经带清单同步的版本。未变化的文件（包括 TensorFlow）不会打进压缩包。

```powershell
.\build_exe.ps1 -Version 1.0.4
.\.venv\Scripts\python.exe workflow_scripts\build_windows_patch.py `
  --manifest-update `
  --baseline dist\baselines\1.0.3 `
  --target dist\E7_BP_Helper `
  --output dist\patches\E7_BP_Helper_update_to_1.0.4.zip
```

每发布一个清单版本，都把 `dist\E7_BP_Helper\` 另存为 `dist\baselines\<版本>\`。下次生成更新时，把所有仍要支持的清单基线都传给 `--baseline`。压缩包只包含这些基线里至少有一份不同的文件。

用户不要解压这个 ZIP。把它放到 `E7_BP_Helper.exe` 同一目录，重新打开程序。程序跳过哈希已经一致的文件，替换不一致的文件，并删除上一份清单里有、新清单里没有的文件。

如果本机某个文件和清单不同，更新包里却没有它，程序会拒绝安装并提示下载完整包。这覆盖文件损坏，以及跳过了某个中间版本的情况。1.0.2 及更早的程序会忽略 `E7_BP_Helper_update_*.zip`。

以下补丁会被拒绝，程序仍会正常启动：

- 压缩包损坏、哈希不一致或包含 `..` 路径：移到 `patches_rejected\`，原因写在同名 `.reason.txt`。
- 补丁的起始版本不是当前已安装版本：ZIP 留在原处，等安装到对应版本后再处理。
- 标记为完整更新的补丁：不会安装。

失败时，更新任务会恢复本次已替换的文件，并把原因写入 `windows_patch_error.log`。

## 版本位置

已安装版本优先读取 EXE 同目录的 `windows_version.json`，没有时读取 `_internal\windows_version.json`。本地服务的 `/api/status` 会返回 `windows_version`。
