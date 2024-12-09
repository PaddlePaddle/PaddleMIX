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

import platform
import traceback

is_main_process = True


def main_process_print(self, *args, sep=" ", end="\n", file=None):
    if is_main_process:
        print(self, *args, sep=sep, end=end, file=file)


def chunked_worker_run(map_func, args, results_queue=None):
    for a in args:
        # noinspection PyBroadException
        try:
            res = map_func(*a)
            results_queue.put(res)
        except KeyboardInterrupt:
            break
        except Exception:
            traceback.print_exc()
            results_queue.put(None)


def chunked_multiprocess_run(map_func, args, num_workers, q_max_size=1000):
    num_jobs = len(args)
    if num_jobs < num_workers:
        num_workers = num_jobs

    queues = [Manager().Queue(maxsize=q_max_size // num_workers) for _ in range(num_workers)]
    if platform.system().lower() != "windows":
        process_creation_func = get_context("spawn").Process
    else:
        process_creation_func = Process

    workers = []
    for i in range(num_workers):
        worker = process_creation_func(
            target=chunked_worker_run, args=(map_func, args[i::num_workers], queues[i]), daemon=True
        )
        workers.append(worker)
        worker.start()

    for i in range(num_jobs):
        yield queues[i % num_workers].get()

    for worker in workers:
        worker.join()
        worker.close()
