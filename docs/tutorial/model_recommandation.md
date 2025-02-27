# 热门任务和推荐模型

## 多模态理解

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
  




