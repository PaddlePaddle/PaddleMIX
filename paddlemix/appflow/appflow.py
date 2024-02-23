# coding:utf-8
# Copyright (c) 2023  PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"
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

import paddle

from paddlemix.utils.tools import get_env_device

from .configuration import APPLICATIONS


class Appflow(object):
    """
    Args:
        app (str): The app name for the Appflow, and get the task class from the name.
        model (str, optional): The model name in the task, if set None, will use the default model.
        mode (str, optional): Select the mode of the task, only used in the tasks of word_segmentation and ner.
            If set None, will use the default mode.
        device_id (int, optional): The device id for the gpu, xpu and other devices, the default value is 0.
        kwargs (dict, optional): Additional keyword arguments passed along to the specific task.

    """

    def __init__(self, app, models=None, mode=None, device_id=0, from_hf_hub=False, **kwargs):
        assert app in APPLICATIONS, f"The task name:{app} is not in Taskflow list, please check your task name."
        self.app = app
        # Set the device for the task
        device = get_env_device()
        if device == "cpu" or device_id == -1:
            paddle.set_device("cpu")
        else:
            paddle.set_device(device + ":" + str(device_id))

        tag = "models"
        ind_tag = "model"
        self.models = models
        if isinstance(self.models, list) and len(self.models) > 0:
            for model in self.models:
                assert model in set(APPLICATIONS[app][tag].keys()), f"The {tag} name: {model} is not in task:[{app}]"
        else:
            self.models = [APPLICATIONS[app]["default"][ind_tag]]

        self.task_instances = []
        for model in self.models:
            if "task_priority_path" in APPLICATIONS[self.app][tag][model]:
                priority_path = APPLICATIONS[self.app][tag][model]["task_priority_path"]
            else:
                priority_path = None

            # Update the task config to kwargs
            config_kwargs = APPLICATIONS[self.app][tag][model]
            kwargs["device_id"] = device_id
            kwargs.update(config_kwargs)
            task_class = APPLICATIONS[self.app][tag][model]["task_class"]
            self.task_instances.append(
                task_class(
                    model=model,
                    task=self.app,
                    priority_path=priority_path,
                    from_hf_hub=from_hf_hub,
                    **kwargs,
                )
            )

        app_list = APPLICATIONS.keys()
        Appflow.app_list = app_list

    def __call__(self, **inputs):
        """
        The main work function in the appflow.
        """
        results = inputs
        for task_instance in self.task_instances:
            # Get input results and put into outputs
            results = task_instance(results)
        return results

    def help(self):
        """
        Return the task usage message.
        """
        return self.task_instance.help()

    def task_path(self):
        """
        Return the path of current task
        """
        return self.task_instance._task_path

    @staticmethod
    def tasks():
        """
        Return the available task list.
        """
        task_list = list(TASKS.keys())  # noqa
        return task_list
