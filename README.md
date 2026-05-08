# 简介
* 此仓库为c++实现, 大体改自[rknpu2](https://github.com/rockchip-linux/rknpu2), python快速部署见于[rknn-multi-threaded](https://github.com/leafqycc/rknn-multi-threaded)
* 使用[线程池](https://github.com/senlinzhan/dpool)异步操作rknn模型, 提高rk3588/rk3588s的NPU使用率, 进而提高推理帧数
* [yolov5s](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)使用relu激活函数进行优化,提高推理帧率

# 更新说明
* 修复了cmake找不到pthread的问题
* 新增nosigmoid分支,使用[rknn_model_zoo](https://github.com/airockchip/rknn_model_zoo/tree/main/models)下的模型以达到极限性能提升
* 将RK3588 NPU SDK 更新至官方主线1.5.0, [yolov5s-silu](https://github.com/rockchip-linux/rknn-toolkit2/tree/v1.4.0/examples/onnx/yolov5)将沿用1.4.0的旧版本模型, [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)更新至1.5.0版本, 弃用nosigmoid分支。
* 新增v1.5.0分支(向下兼容1.4.0), main分支更新至v1.5.2, 修改了项目结构, 将rknn模型线程池封装成类(include/rknnPool.hpp)

# 使用说明
### 演示
  * 系统需安装有**OpenCV**
  * 下载Releases中的测试视频于项目根目录,运行build-linux_RK3588.sh
  * 可切换至root用户运行performance.sh定频提高性能和稳定性
  * 编译完成后进入install运行命令./rknn_yolov5_demo **模型所在路径** **视频所在路径/摄像头序号**

### 部署应用
  * 参考include/rkYolov5s.hpp中的rkYolov5s类构建rknn模型类

# 多线程模型帧率测试
* 使用performance.sh进行CPU/NPU定频尽量减少误差
* 测试模型来源: 
* [yolov5s-relu](https://github.com/rockchip-linux/rknpu2/tree/master/examples/rknn_yolov5_demo/model/RK3588)
* 测试视频可见于 [bilibili](https://www.bilibili.com/video/BV1zo4y1x7aE/?spm_id_from=333.999.0.0)

|  模型\线程数   | 1    |  2   | 3  |  4  | 5  | 6  | 9  | 12  |
|  ----  | ----  |  ----  | ----  |  ----  | ----  | ----  | ----  | ----  |
| Yolov5s - relu  | 41.6044 | 71.6037 | 98.6057 | 98.0068 | 104.6001 | 114.7454 | 129.5693 | 140.8788 |

# 补充
* 异常处理尚未完善, 目前仅支持rk3588/rk3588s下的运行

# Acknowledgements
* https://github.com/rockchip-linux/rknpu2
* https://github.com/senlinzhan/dpool
* https://github.com/ultralytics/yolov5
* https://github.com/airockchip/rknn_model_zoo


# SVM Detection Dataset — Build Story

> **Goal:** train a vehicle Surround-View Monitoring (SVM) detector that recognises four road-user classes: **Pedestrian · Bicycle · Motorcycle · Personal Mobility (PM)** — to run on an on-board NPU (OrangePi 5 Plus / RK3588).
> **Latest snapshot:** 105,922 images · 359,711 labelled objects.

---

## 1. How the dataset grew

```mermaid
flowchart LR
    A[COCO public dataset<br/>kept only target classes] --> B[Synthetic SVM composites<br/>place objects on real SVM frames]
    B --> C[Add human-annotated<br/>real SVM frames]
    C --> D[Add on-road recordings<br/>4 classes labelled by hand]
    D --> E[Add background images<br/>to teach the model what to ignore]
    E --> F[Add a session rich in<br/>scooters / e-mobility]
    F --> G[(Final dataset)]
```

We did not collect everything at once. The dataset was built in stages — each stage targeted a specific weakness we observed in the model after testing.

---

## 2. The journey, version by version

```mermaid
flowchart LR
    S1[Stage 1<br/>73k imgs<br/>COCO target classes<br/>+ first SVM composites] --> S2
    S2[Stage 2<br/>87k imgs<br/>more synthetic SVM scenes] --> S3
    S3[Stage 3<br/>180k imgs<br/>real on-road recordings<br/>+ early PM examples] --> S4
    S4[Stage 4<br/>100k imgs<br/>curated subset<br/>+ first backgrounds] --> S5
    S5[Stage 5<br/>105k imgs<br/>hand-reviewed PM session<br/>+ more backgrounds]
```

| Stage | Total images | Why this stage was added |
|---|---|---|
| 1 | 73,057  | First working baseline — recognise the four classes at all |
| 2 | 87,723  | Improve coverage of poses and viewpoints |
| 3 | 180,658 | Bring in real on-road footage + first scooter / PM examples |
| 4 | 100,552 | Curated subset + first batch of "do-not-trigger" backgrounds |
| 5 (current) | **105,922** | Targeted scooter session + more curated backgrounds |

> Note on the image-count drop between Stage 3 and Stage 4: we removed redundant synthetic frames and switched to a leaner, fully-symlinked dataset that takes ~800 MB on disk instead of 89 GB, while keeping the high-value examples.

---

## 3. What's inside the current dataset

```mermaid
pie title Image sources
    "Synthetic SVM + COCO subset" : 90329
    "Human-annotated SVM" : 4629
    "Background frames" : 8068
    "PM-rich session" : 2896
```

| Category | Images | Role |
|---|---|---|
| Synthetic SVM composites + COCO subset | 90,329 | Bulk class-recognition signal |
| Human-annotated SVM / on-road frames    |  4,629 | Real-world appearance and geometry |
| Background frames (no objects)          |  8,068 | Suppress false alarms |
| PM-rich hand-reviewed session           |  2,896 | Strengthen the rarest class |

---

## 4. Class distribution

| Class | Total bounding boxes | Share |
|---|---|---|
| Pedestrian | **322,207** | 89.6 % |
| Bicycle    | **13,210**  | 3.7 % |
| Motorcycle | **12,523**  | 3.5 % |
| PM         | **11,771**  | 3.3 % |

![Class distribution](fig_class_split.png)

Pedestrians dominate naturally — they are the most frequent road user. The other three classes are rarer but consistently present, and we have spent significant effort to keep them well-represented.

---

## 5. Where objects appear in the image

The image format is a **640 × 640 stacked composite** of two camera views (top + bottom). The heatmap below shows where labelled objects sit. The two horizontal bands correspond to the two stacked camera halves.

![Annotation location heatmap](fig_centers.png)

This visualisation confirms the camera geometry is consistent across the whole dataset.

---

## 6. Image sizes and formats

- **Canonical input size:** 640 × 640 (the SVM stack)
- Some legacy COCO-derived images come in other sizes; the training pipeline standardises everything to 640 px automatically.
- **Format:** JPEG 98 %, PNG 2 %.

![Image format breakdown](fig_format.png)

---

## 7. Object size and density

- Most labelled objects are **small to medium** in pixel size — consistent with road-users seen from the vehicle's surround cameras.
- A typical image contains **0–6 objects**. Some frames intentionally contain **zero objects** (the background-only frames).

![Bounding-box size distribution](fig_bbox_size.png)
![Objects per image](fig_obj_per_img.png)

---

## 8. Train / Validation split

```mermaid
pie title Train and Val split
    "Train" : 90034
    "Val" : 15888
```

The split is a fixed random shuffle (reproducible). Background-only images are spread across both splits in the same proportion.

---

## 9. Why each stage was added — short summary

1. **Stage 1–2 — COCO + Synthetic SVM:** start with public, well-labelled imagery and combine it with our SVM camera scenes so the model first learns what each class looks like.
2. **Stage 3 — Real on-road recordings:** the synthetic composites cannot capture real-world lighting, motion blur and camera distortion. Adding hand-labelled drives gives the model authentic context.
3. **Stage 4 — Background images:** when we tested the model on real driving footage we saw it "imagining" objects (false alarms) on poles, trees and shadows. We collected those exact scenes and labelled them as empty so the model learns to ignore them.
4. **Stage 5 — PM-rich session + more backgrounds:** PM (e-scooters / e-mobility) is the rarest class and was being missed. We recorded a dedicated outing where PM riders were present, then a human reviewed every frame to ensure correct labels.

---

## 10. Where this is leading

Each stage of the dataset has produced a measurable improvement in the model. The current model:
- recognises **scooters / PM far more reliably** than before;
- **almost never produces false alarms** on plain-background driving scenes;
- runs on the on-board NPU within the project's real-time budget.

We continue to mine new failure modes from new recordings; those become the next stage of the dataset.
