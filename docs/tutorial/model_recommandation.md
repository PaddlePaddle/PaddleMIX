# 热门任务和推荐模型

## 多模态理解

### 多模态理解效果示例如下：

<img src="https://github.com/user-attachments/assets/4c9a0427-57c7-4e1b-80f0-428c03119cc3"></img>


多模态理解🤝融合了视觉👀和语言💬处理能力。包含基础感知、细粒度图像理解和复杂视觉推理🧠等功能。这些技术可应用于教育📚、医疗🏥、工业🏭等多个领域，实现从静态图像🖼️到动态视频🎥的全面智能分析。


#### **1. 视觉问答（Visual Question Answering, VQA）**
**任务描述**：基于图像或视频内容，回答自然语言问题，需同时理解视觉语义、空间关系及常识知识。核心挑战在于跨模态对齐精度和事实性推理能力，需避免生成与图像无关的"幻觉答案"。
**关注能力**：
- 细粒度视觉理解（物体属性、空间关系）
- 跨模态语义对齐（视觉特征与文本问题的关联）
- 常识与专业领域知识内化

**推荐模型**：
- [**Qwen2.5-VL**](../../paddlemix/examples/qwen2_5_vl/README.md)
- [**LLaVA-OneVision**](../../paddlemix/examples/llava_onevision/README.md)
- [**InternVL2**](../../paddlemix/examples/internvl2/README.md)

---

#### **2. 文献和图表理解（Document and Diagrams Reading）**
**任务描述**：解析PDF/扫描文档、表格、科学图表等结构化数据，提取关键信息并执行推理。需处理复杂排版、手写体、数学符号等特殊元素。
**关注能力**：
- 任意分辨率文本识别（OCR）
- 表格结构重建与跨单元格推理
- 数学公式/化学式语义解析

**推荐模型**：
- [**PP-DocBee**](../../paddlemix/examples/ppdocbee/README.md)
- [**Qwen2.5-VL**](../../paddlemix/examples/qwen2_5_vl/README.md)
- [**Aria**](../../paddlemix/examples/aria/README.md)

---

#### **3. 数学推理（Mathematical Reasoning）**
**任务描述**：结合文本、公式、图表等多模态信息解决数学问题，需执行符号运算、几何证明等复杂推理流程。
**关注能力**：
- 多模态条件解析（将图表数据转化为数学表达式）
- 分步逻辑链生成与验证
- 符号计算与数值精度控制

**推荐模型**：
- [**DeepSeek-VL2**](../../paddlemix/examples/deepseek_vl2/README.md)
- [**Qwen2.5-VL**](../../paddlemix/examples/qwen2_5_vl/README.md)

---

#### **4. 指示性目标检测（Referring Expression Comprehension）**
**任务描述**：根据自然语言指令定位并检测中图像/视频中的特定目标，返回box坐标，需理解抽象描述（如"左起第三个穿红衣服的人"）。
**关注能力**：
- 开放词汇实例分割
- 空间关系推理（方位词、序数词理解）
- 跨帧一致性保持

**推荐模型**：
- [**Qwen2-VL**](../../paddlemix/examples/qwen2_vl/README.md)
- [**DeepSeek-VL2**](../../paddlemix/examples/deepseek_vl2/README.md)

---

#### **5. 视频理解（Video Understanding）**
**任务描述**：解析长视频（数十分钟至数小时）中的时序事件、人物交互、场景变换，需捕捉时空动态特征。

**关注能力**：
- 动态分辨率帧采样
- 跨镜头事件关联
- 秒级时间戳定位

**推荐模型**：
- [**Qwen2.5-VL**](../../paddlemix/examples/qwen2_5_vl/README.md)
- [**LLaVA-OneVision**](../../paddlemix/examples/llava_onevision/README.md)
- [**InternVL2**](../../paddlemix/examples/internvl2/README.md)


---

#### **6. 视觉Agent**
**任务描述**：构建可操作物理世界/数字界面的智能体，完成点击、拖拽等具体动作。
**关注能力**：
- 屏幕元素OCR与操作映射
- 多步骤任务规划
- 异常状态恢复

**推荐模型**：
- [**Qwen2.5-VL**](../../paddlemix/examples/qwen2_5_vl/README.md)
- [**LLaVA-OneVision**](../../paddlemix/examples/llava_onevision/README.md)


## 多模态生成

### 多模态生成效果示例如下：
<div style="display: flex; justify-content: center; gap: 5px;">
    <img src="https://github.com/user-attachments/assets/f4768f08-f7a3-45e0-802c-c91554dc5dfc" style="height: 250px; object-fit: fill;">
    <img src="https://github.com/user-attachments/assets/9bf4a333-af57-4ddd-a514-617dea8da435" style="height: 250px; object-fit: fill;">
</div>


多模态生成✍️融合了文本💬与视觉👀的创造能力。涵盖了从文字生成图像🖼️到文字生成视频🎥的各类技术。功能涉及艺术创作🎨、动画制作📽️、内容生成📝等。可以在教育📚、娱乐🎮、广告📺等领域实现从静态图像到动态视频的创意生成。


#### **1. 文本图像生成 (Text-to-Image)**
**任务描述**：根据用户提供的自然语言描述（文本），生成符合语义要求且视觉上合理、高质量、多样化的图像。应用于艺术设计、广告创意等领域。

**关注能力**：
- 高质量图像生成
- 多样性与可控性
- 文本+图像的可控生成


**推荐模型**：
- [**Stable Diffusion 3**](../../ppdiffusers/examples/text_to_image/README_sd3.md)
- [**ControlNet**](../../ppdiffusers/examples/controlnet/README.md)

---

#### **2. 文本视频生成 (Text-to-Video)**
**任务描述**：通过自然语言描述生成符合语义的动态视频内容的技术。其核心目标是实现‌跨模态对齐‌（文本到视觉的映射）和‌时序连续性‌（视频帧间的动态连贯），生成高质量、可控且符合用户意图的视频。可用于内容创作、虚拟现实等领域。

**关注能力**：
- 时序建模能力
- 视觉保真度与多样性



**推荐模型**：
- [**Cogvideo**](../../ppdiffusers/examples/cogvideo/README.md)
- [**Open-Sora**](../../ppdiffusers/examples/Open-Sora/README.md)


---

#### **3. 视频可控制生成**
**任务描述**：通过输入自然语言描述和控制信号（包括关键点、边缘检测、mask），生成符合与文本、可控信号对齐的视频。可用于内容创作、视频编辑等领域。

**关注能力**：
- 多样性与可控性
- 跨模态能力对齐



**推荐模型**：
- [**ppvctrl**](../../ppdiffusers/examples/ppvctrl/README_CN.md)
- [**AnimateAnyone**](../../ppdiffusers/examples/AnimateAnyone/README.md)
