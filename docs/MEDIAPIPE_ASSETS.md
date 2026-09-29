# 前端姿态资产（MediaPipe）

浏览器端姿态估计依赖以下 MediaPipe 资产。它们**不是本项目的代码**，版权归 Google（Apache-2.0），随仓库分发时请遵守其许可证。

## 需要的文件

放在项目根目录的 `mediapipe/` 下：

| 文件 | 大小(约) | 说明 |
|---|---|---|
| `pose_landmarker_lite.task` | 5.7 MB | 姿态关键点模型（lite） |
| `vision_bundle.mjs` | 137 KB | MediaPipe Tasks-Vision 的 ES Module 打包文件 |
| `vision_wasm_internal.js` | 205 KB | SIMD 版 wasm 加载器 |
| `vision_wasm_internal.wasm` | 9.4 MB | SIMD 版 wasm 运行时 |
| `vision_wasm_nosimd_internal.js` | 205 KB | 非 SIMD 版加载器 |
| `vision_wasm_nosimd_internal.wasm` | 9.3 MB | 非 SIMD 版运行时 |

## 前端如何引用

`index.html` 中的模块脚本：

```js
const { FilesetResolver, PoseLandmarker } = await import('/mediapipe/vision_bundle.mjs');
const vision = await FilesetResolver.forVisionTasks('/mediapipe');   // 注意：不带结尾斜杠
window.__mpPoseLandmarker = await PoseLandmarker.createFromOptions(vision, {
  baseOptions: { modelAssetPath: '/mediapipe/pose_landmarker_lite.task', delegate: 'GPU' },
  runningMode: 'VIDEO',
  numPoses: 1,
  /* ... */
});
```

> `FilesetResolver.forVisionTasks(base)` 会拼接 `${base}/vision_wasm_internal.(js|wasm)`，因此 `base` 传 `/mediapipe`（不要带结尾斜杠）。

## Web 服务器要求（重要）

浏览器会因 MIME 类型不正确而拒绝加载。请确保：

| 扩展名 | 需返回的 `Content-Type` |
|---|---|
| `.mjs` | `application/javascript` |
| `.wasm` | `application/wasm` |
| `.task` | `application/octet-stream` 即可（前端用 `fetch` 取二进制） |

nginx 示例片段：

```nginx
location /mediapipe/ {
    alias /path/to/MoonWave/mediapipe/;
    types {
        application/javascript mjs;
        application/wasm       wasm;
    }
    default_type application/octet-stream;
}
```

## 从哪里获取

这些资产来自 Google 的 MediaPipe Tasks-Vision：

1. **`vision_bundle.mjs` 与 wasm 运行时**：来自 npm 包 `@mediapipe/tasks-vision`（解包后 `vision_bundle.mjs` 及 `wasm/` 目录下的 `vision_wasm_internal.*`、`vision_wasm_nosimd_internal.*`）。
2. **姿态模型 `pose_landmarker_lite.task`**：MediaPipe Pose Landmarker 的模型文件（官方模型库 / Tasks-Vision 模型页提供）。

> 建议**自托管**这些文件（放在本站 `mediapipe/` 下），而不是依赖第三方 CDN，以避免网络不可达导致姿态功能失效。

## 回退机制

若浏览器端无法加载上述资产（或初始化失败），前端会自动回退到**逐帧上传图片 → 服务端 MediaPipe 推理**的方式。此时需要服务端也安装并可运行 `mediapipe`（见 `requirements.txt`）。
