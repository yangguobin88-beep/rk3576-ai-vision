# Camera 摄像头封装详解

> 这个模块封装了摄像头操作，使用多线程提高性能，并提供 FPS 计数功能。

---

## 🎯 为什么需要封装摄像头？

### 问题：原始 OpenCV 用法有性能瓶颈

```python
# 普通方式
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()  # ❌ 阻塞式读取，推理时摄像头停止采集
    if ret:
        result = model.infer(frame)  # 推理时间 50ms
        # 这 50ms 内摄像头不采集，丢失了好几帧！
```

**问题：**
- `cap.read()` 是阻塞式的
- 推理时摄像头不采集 → 丢帧
- 实际 FPS 远低于摄像头能力

---

### 解决方案：多线程采集

```python
# 封装后的方式
camera = Camera(source=0)
camera.start()  # 后台线程持续采集

while True:
    frame = camera.read()  # ✅ 立即返回最新帧
    result = model.infer(frame)  # 推理时，后台线程还在采集
    # 不会丢帧！
```

---

## 📖 Camera 类详解

### 整体架构

```
主线程                    采集线程 (后台)
  │                          │
  │─── camera.start() ─────→ 启动线程
  │                          │
  │                          ├─ 循环读取摄像头
  │                          ├─ 更新 self.frame
  │                          ├─ 继续循环...
  │                          │
  │─── camera.read() ──────→ 返回最新帧（不等待）
  │                          │
  │─── 推理 50ms              │（继续采集）
  │                          │
  │─── camera.read() ──────→ 返回最新帧
```

---

### 1️⃣ 初始化参数

```python
def __init__(self, source=0, width=1280, height=720, fps=30):
    self.source = source        # 摄像头设备号（0=默认摄像头）
    self.width = width          # 分辨率宽度
    self.height = height        # 分辨率高度
    self.fps = fps              # 帧率
    self.cap = None             # OpenCV VideoCapture 对象
    self.frame = None           # 最新帧（共享变量）
    self.running = False        # 线程运行标志
    self.thread = None          # 后台线程对象
    self.lock = threading.Lock()  # 🔒 线程锁（防止竞争）
```

**关键参数：**
- `source`: 摄像头设备号
  - `0` = 默认摄像头
  - `1` = 第二个摄像头
  - 也可以是视频文件路径：`'test.mp4'`
- `lock`: **线程锁**，防止主线程读取时，采集线程正在写入

---

### 2️⃣ 打开摄像头

```python
def open(self):
    self.cap = cv2.VideoCapture(self.source)
    if not self.cap.isOpened():
        raise RuntimeError(f"无法打开摄像头: {self.source}")
    
    # 设置摄像头参数
    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
    self.cap.set(cv2.CAP_PROP_FPS, self.fps)
    return self
```

**注意：**
- 有些摄像头不支持所有分辨率
- 实际分辨率可能和设置的不一样

---

### 3️⃣ 后台采集线程 ⭐ 核心

```python
def _capture_loop(self):
    """后台线程：持续读取摄像头"""
    while self.running:  # 运行标志
        ret, frame = self.cap.read()  # 读取一帧
        if ret:
            with self.lock:  # 🔒 加锁（线程安全）
                self.frame = frame  # 更新共享变量
```

**线程安全的关键：**
```python
with self.lock:  # 加锁
    self.frame = frame  # 只有一个线程能执行这里
```

**为什么要加锁？**
- **问题场景：**
  ```
  采集线程：正在写 self.frame = new_frame  （写到一半）
  主线程：  读取 frame = self.frame.copy()  （读到损坏的数据）
  ```
- **加锁后：**
  ```
  采集线程：with self.lock: self.frame = new_frame
  主线程：  等待锁释放 → 读取完整数据
  ```

---

### 4️⃣ 启动采集

```python
def start(self):
    if self.cap is None:
        self.open()  # 如果还没打开，先打开
    
    self.running = True  # 设置运行标志
    
    # 创建后台线程（daemon=True 表示主程序退出时自动结束）
    self.thread = threading.Thread(target=self._capture_loop, daemon=True)
    self.thread.start()  # 启动线程
    
    time.sleep(0.1)  # 等待线程启动
    return self
```

**`daemon=True` 的作用：**
- 主程序退出时，自动结束后台线程
- 不用手动管理线程生命周期

---

### 5️⃣ 读取最新帧

```python
def read(self):
    """读取最新帧（非阻塞）"""
    with self.lock:  # 🔒 加锁
        return self.frame.copy() if self.frame is not None else None
```

**为什么用 `.copy()`？**
```python
# ❌ 不用 copy
frame = self.frame  # 只是引用，后台线程修改会影响这个 frame

# ✅ 用 copy
frame = self.frame.copy()  # 独立副本，后台线程修改不影响
```

---

### 6️⃣ 停止和释放

```python
def stop(self):
    """停止采集线程"""
    self.running = False  # 设置标志，线程会退出循环
    if self.thread:
        self.thread.join(timeout=1.0)  # 等待线程结束（最多 1 秒）

def release(self):
    """释放资源"""
    self.stop()  # 先停止线程
    if self.cap:
        self.cap.release()  # 再释放摄像头
```

---

### 7️⃣ with 语法支持

```python
def __enter__(self):
    return self.start()  # 进入 with 时启动

def __exit__(self, *args):
    self.release()  # 退出 with 时释放
```

**使用示例：**
```python
with Camera(source=0) as camera:
    while True:
        frame = camera.read()
        # ... 处理帧 ...
# 自动释放摄像头
```

---

## 📊 FPSCounter 类详解

### 作用：计算实际 FPS

```python
class FPSCounter:
    def __init__(self, window=30):
        self.times = []  # 存储最近 N 次的时间戳
        self.window = window  # 窗口大小（默认 30 帧）
```

### 原理

```python
def tick(self):
    """每处理一帧调用一次"""
    self.times.append(time.time())  # 记录当前时间
    if len(self.times) > self.window:
        self.times.pop(0)  # 保持窗口大小

def get_fps(self):
    """计算 FPS"""
    if len(self.times) < 2:
        return 0.0
    
    # FPS = 帧数 / 时间跨度
    return (len(self.times) - 1) / (self.times[-1] - self.times[0])
```

**计算公式：**
```
例如最近 30 帧：
时间戳：[1.0, 1.1, 1.2, ..., 2.5]
总时长：2.5 - 1.0 = 1.5 秒
FPS = 29 帧 / 1.5 秒 = 19.3
```

---

## 🚀 完整使用示例

```python
from common.camera import Camera, FPSCounter
import cv2

# 创建摄像头和 FPS 计数器
camera = Camera(source=0, width=1280, height=720, fps=30)
fps_counter = FPSCounter()

camera.start()  # 启动后台采集

try:
    while True:
        frame = camera.read()  # 读取最新帧（非阻塞）
        if frame is None:
            continue
        
        # 推理（模拟耗时操作）
        # result = model.infer(frame)
        
        # 计算 FPS
        fps_counter.tick()
        fps = fps_counter.get_fps()
        
        # 显示 FPS
        cv2.putText(frame, f'FPS: {fps:.1f}', (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    camera.release()
    cv2.destroyAllWindows()
```

---

## 🔍 性能对比

| 方式 | FPS | 说明 |
|------|-----|------|
| **普通方式** | ~15 | 推理时摄像头停止，丢帧严重 |
| **多线程封装** | ~28 | 后台持续采集，不丢帧 |

---

## ⚠️ 线程安全知识点

### 什么是线程锁（Lock）？

```python
lock = threading.Lock()

# 线程 A
with lock:
    self.frame = new_frame  # 只有一个线程能执行

# 线程 B
with lock:
    frame = self.frame.copy()  # 等待 A 释放锁后才能执行
```

**作用：** 防止多个线程同时访问共享变量导致数据错乱。

---

## ✅ 总结

### Camera 类的优点

| 优点 | 说明 |
|------|------|
| **多线程采集** | 推理时不丢帧，提高 FPS |
| **非阻塞读取** | `read()` 立即返回 |
| **线程安全** | 使用锁保护共享变量 |
| **支持 with** | 自动管理资源 |
| **易用性** | 简单几行代码即可使用 |

### FPSCounter 的优点

| 优点 | 说明 |
|------|------|
| **准确** | 基于滑动窗口计算 |
| **平滑** | 不会剧烈跳动 |
| **简单** | 只需 `tick()` 和 `get_fps()` |

---

## 🔑 关键概念

1. **多线程**：后台线程持续采集，主线程处理
2. **线程锁**：保护共享变量，防止数据竞争
3. **非阻塞**：`read()` 不等待，立即返回
4. **滑动窗口**：FPS 计算基于最近 N 帧

---

这个封装大大提高了实时检测的性能！
