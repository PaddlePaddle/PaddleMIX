## 1. DiffSinger 模型介绍

该模型是 [DiffSinger（由 OpenVPI 维护的版本）](https://github.com/openvpi/DiffSinger) 的 Paddle 实现。

[DiffSinger](https://arxiv.org/abs/2105.02446) 是目前最先进的歌声合成（Singing Voice Synthesis, SVS）模型。OpenVPI 维护的版本对其进行了进一步优化，增加了更多的功能。

本仓库目前仅支持推理功能，后续会对该仓库进一步完善说明，


## 2. 环境准备

1）[安装 PaddleMix 环境依赖包](https://github.com/PaddlePaddle/PaddleMIX/tree/b4f97ff859e1964c839fc5fab94f7ba63b1e5959?tab=readme-ov-file#%E5%AE%89%E8%A3%85)

3）使用 pip 安装依赖:
   ```bash
   pip install -r requirements.txt
   ```

## 4. 快速开始
1）完成环境准备后，下载权重至`PaddleMIX/paddlemix/examples/diffsinger/openvpi`
```bash
cd paddlemix/examples/diffsinger/

wget https://paddlenlp.bj.bcebos.com/models/community/paddlemix/openvpi.tar

tar -xvf openvpi.tar 

```

然后运行以下脚本：

```bash
bash run_predict.sh
```


## 5. Demo
点击下载音频进行试听～
<div align = "center">
  <thead>
  </thead>
  <tbody>
   <tr>
      <td align = "center">
      <a href="https://paddlenlp.bj.bcebos.com/models/community/paddlemix/audio/00_我多想说再见啊.wav" rel="nofollow">
            <img align="center" src="https://user-images.githubusercontent.com/20476674/209344877-edbf1c24-f08d-4e3b-88a4-a27e1fd0a858.png" width="200 style="max-width: 100%;"></a><br>
      </td>
    </tr>
  </tbody>
</div>
</details>