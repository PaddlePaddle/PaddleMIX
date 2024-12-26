# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import multiprocessing


class PaddleXLayoutParser(object):
    def __init__(self, gpu_id=0):
        self.gpu_id = gpu_id
        self._pipeline = None
        
    @property
    def default_pipeline(self, ):
        if self._pipeline is None:
            self._pipeline = self.new_pipeline(self.gpu_id)
        return self._pipeline

    def new_pipeline(self, gpu_id):
        from paddlex import create_pipeline
        pipeline = create_pipeline(pipeline="layout_parsing", device=f"gpu:{gpu_id}")
        return pipeline
    
    def process_image(self, image_path, pipeline=None):
        if pipeline is None:
            pipeline = self.default_pipeline

        output = pipeline.predict(image_path)
        try:
            n = 0
            for res in output:
                # res.print()  # 打印预测的结构化输出
                res_json = res.json
                set_img_to_empty(res_json) #不包含图片的OCR版面信息
                return res_json
        except Exception as e:
            print(f"{e}")

    def _process_images(self, gpu_id, image_paths, q):
        pipeline = self.new_pipeline(gpu_id=gpu_id)
        q.put([self.process_image(path, pipeline=pipeline) for path in image_paths])

    def process_images(self, image_paths, num_gpus=1, processes_per_gpu=1):
        # 将图片文件分配到每个GPU和进程
        q=multiprocessing.Queue()
        chunk_size = len(image_paths) // (num_gpus * processes_per_gpu)
        chunks = [image_paths[i:i + chunk_size] for i in range(0, len(image_paths), chunk_size)]

        # 创建多进程
        processes = []
        for gpu_id in range(num_gpus):
            for _ in range(processes_per_gpu):
                if chunks:
                    chunk = chunks.pop(0)
                    p = multiprocessing.Process(target=self._process_images, args=(gpu_id, chunk, q))
                    processes.append(p)
                    p.start()

        # 等待所有进程完成
        for p in processes:
            p.join()

        # 获取结果并返回
        results = [q.get() for p in processes]

        return results

    
def set_img_to_empty(d):
    for k, v in d.items():
        if isinstance(v, dict):
            set_img_to_empty(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, dict):
                    set_img_to_empty(item)
                else:
                    if k == 'img':
                        d[k] = None
        else:
            if k == 'img':
                d[k] = None



if __name__ == '__main__':

    pass

