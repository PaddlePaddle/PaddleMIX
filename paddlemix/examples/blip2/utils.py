# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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
import copy
import datetime
import json
import os
import re
import sys
import time

import paddle
from pycocoevalcap.eval import COCOEvalCap
from pycocotools.coco import COCO

from paddlemix.models.blip2.eva_vit import interpolate_pos_embed
from paddlemix.utils.downloader import get_weights_path_from_url, is_url
from paddlemix.utils.log import logger

LLM_LIST = {
    "facebook/opt-2.7b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/opt-2.7b/model_state.pdparams",
    "t5-small": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-small/model_state.pdparams",
    "t5-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-base/model_state.pdparams",
    "t5-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-large/model_state.pdparams",
    "t5-3b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-3b/model_state.pdparams",
    "t5-11b": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-11b/model_state.pdparams",
    "t5-v1_1-base": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-base/model_state.pdparams",
    "t5-v1_1-large": "https://bj.bcebos.com/paddlenlp/models/transformers/t5/t5-v1_1-large/model_state.pdparams",
    "facebook/llama-7b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-7b/model_state.pdparams",
    "facebook/llama-13b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-13b/model_state.pdparams",
    "facebook/llama-30b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-30b/model_state.pdparams",
    "facebook/llama-65b": "https://bj.bcebos.com/paddlenlp/models/community/facebook/llama-65b/model_state.pdparams",
}


class BlipCollator:
    """
    Data collator that will dynamically pad the inputs to the longest sequence in the batch.

    Args:
        processor (`paddlemix.processors.ProcessorMixin`):
            The processor used for pre-process the data.
    """

    def __init__(self, processor, mode="train"):
        self.processor = processor
        self.mode = mode

    def __call__(self, data_list):
        images = [sample["image"] for sample in data_list]
        if "text_input" not in data_list[0].keys():
            text = None
        else:
            text = [sample["text_input"] for sample in data_list]
        image_id = [sample["image_id"] for sample in data_list]
        batch = self.processor(
            images=images,
            text=text,
            max_length=32,
            return_tensors="pd",
            return_attention_mask=True,
            mode=self.mode,
        )
        batch.update({"image_id": image_id})
        return batch


def coco_caption_eval(coco_gt_root, results_file, split):
    # urls = {
    #     "val": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_val_gt.json",
    #     "test": "https://storage.googleapis.com/sfr-vision-language-research/datasets/coco_karpathy_test_gt.json",
    # }
    filenames = {
        "val": "coco_karpathy_val_gt.json",
        "test": "coco_karpathy_test_gt.json",
    }

    # download_url(urls[split], coco_gt_root)
    annotation_file = os.path.join(coco_gt_root, filenames["test"])

    # create coco object and coco_result object
    coco = COCO(annotation_file)
    coco_result = coco.loadRes(results_file)

    coco_eval = COCOEvalCap(coco, coco_result)
    coco_eval.evaluate()

    # print output evaluation scores
    for metric, score in coco_eval.eval.items():
        print(f"{metric}: {score:.3f}")

    return coco_eval


def load_model(args, model, optimizer=None, ckpt_dir="", load_language_model=True):
    """
    load the saved checkpoint file and update the state dicts of model and optimizer.
    """
    if ckpt_dir is None:
        return

    if not os.path.exists(ckpt_dir):
        ValueError("Cannot find pretrained model path: {}".format(ckpt_dir))

    if os.path.isfile(ckpt_dir):
        path = ckpt_dir
    elif is_url(ckpt_dir):
        path = get_weights_path_from_url(ckpt_dir)
    else:
        assert os.path.exists(ckpt_dir), f"{ckpt_dir} not exist"

    ckpt_dir = path
    if ckpt_dir and os.path.isfile(ckpt_dir):
        # breakpoint()
        print("Try to load a whole checkpoint from %s " % ckpt_dir)
        embedding_list = ["word_embeddings"]
        collinear_list = [
            "fc1",
            "fc2",
            "qkv",
            "proj",
            "query",
            "key",
            "value",
            "qkv_proj",
            "q_proj",
            "k_proj",
            "v_proj",
            "linear1",
            "linear2",
            "project_in",
            "project_out",
        ]
        rowlinear_list = ["out_proj"]
        all_list = collinear_list + rowlinear_list + embedding_list
        skip_list = ["visual_encoder.patch_embed.proj.weight", "visual_encoder.patch_embed.proj.bias"]

        col_list = []
        row_list = []
        emb_list = []

        mp_rank = args.mp_rank
        mp_size = args.tensor_parallel_degree

        def renamebias(model_dict, whole_key):
            if "q_bias" in whole_key:
                key = whole_key.replace("q_bias", "q_proj.bias")
            elif "v_bias" in whole_key:
                key = whole_key.replace("v_bias", "v_proj.bias")
            model_dict[key] = model_dict[whole_key]
            del model_dict[whole_key]
            return model_dict

        def col_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[1] // mp_size
                return model_dict[:, mp_rank * subbatch : (mp_rank + 1) * subbatch]
            elif len(model_dict.shape) == 1:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        def row_split_modeldict(model_dict):
            if len(model_dict.shape) == 2:
                subbatch = model_dict.shape[0] // mp_size
                return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]
            else:
                return model_dict

        def emb_split_modeldict(model_dict):
            subbatch = model_dict.shape[0] // mp_size
            return model_dict[mp_rank * subbatch : (mp_rank + 1) * subbatch]

        model_dict = paddle.load(ckpt_dir)
        for whole_key in model_dict.keys():
            if "." not in whole_key:
                continue

            key = whole_key.split(".")[-2]
            if whole_key in skip_list:
                continue
            if key in all_list:
                if key in collinear_list:
                    col_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = col_split_modeldict(model_dict[whole_key])
                elif key in rowlinear_list:
                    row_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = row_split_modeldict(model_dict[whole_key])
                else:
                    emb_list.append((key, model_dict[whole_key].shape))
                    model_dict[whole_key] = emb_split_modeldict(model_dict[whole_key])

        param_state_dict = model_dict
        import numpy as np

        model_dict = model.state_dict()
        model_weight = {}
        incorrect_keys = 0
        for key, value in model_dict.items():
            if key in param_state_dict.keys():

                if isinstance(param_state_dict[key], np.ndarray):
                    param_state_dict[key] = paddle.to_tensor(param_state_dict[key])
                if value.dtype == param_state_dict[key].dtype:
                    model_weight[key] = param_state_dict[key]
                else:
                    model_weight[key] = param_state_dict[key].astype(value.dtype)
                if value.shape != param_state_dict[key].shape:
                    logger.info("Unmatched key: {}".format(key))
                    print(value.shape, param_state_dict[key].shape, key)

            else:
                if load_language_model is False and "language_model" in key:
                    continue
                logger.info("Unmatched key: {}".format(key))
                incorrect_keys += 1
        interpolate_pos_embed(model, model_weight)
        model.set_state_dict(model_weight)

        del model_dict
    else:
        print("`load` requires a valid value of `ckpt_dir`.")
        raise TypeError("`load` requires a valid value of `ckpt_dir`.")


def save_result(result, result_dir, filename, remove_duplicate="", world_size=1):
    import logging

    rank_id_curr_node = int(os.environ.get("PADDLE_RANK_IN_NODE", 0))
    result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank_id_curr_node))
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    json.dump(result, open(result_file, "w"))

    final_result_file = os.path.join(result_dir, "%s.json" % filename)
    if world_size > 1:
        paddle.distributed.barrier()
    if rank_id_curr_node == 0:
        logging.warning("rank %d starts merging results." % rank_id_curr_node)
        result = []
        # for rank in range(get_world_size()):
        for rank in range(int(os.environ.get("PADDLE_TRAINERS_NUM", 1))):
            result_file = os.path.join(result_dir, "%s_rank%d.json" % (filename, rank))
            res = json.load(open(result_file, "r"))
            result += res

        if remove_duplicate:
            result_new = []
            id_list = []
            for res in result:
                if res[remove_duplicate] not in id_list:
                    id_list.append(res[remove_duplicate])
                    result_new.append(res)
            result = result_new

        json.dump(result, open(final_result_file, "w"))
        print("result file saved to %s" % final_result_file)
    else:
        while not os.path.exists(final_result_file):
            time.sleep(0.5)
            logging.warning("rank %d waits rank0 to merge results." % rank_id_curr_node)

    # combine results from all processes
    return final_result_file


class VQAEval:
    def __init__(self, vqa=None, vqaRes=None, n=2):
        self.n = n
        self.accuracy = {}
        self.evalQA = {}
        self.evalQuesType = {}
        self.evalAnsType = {}
        self.vqa = vqa
        self.vqaRes = vqaRes
        if vqa is not None:
            self.params = {"question_id": vqa.getQuesIds()}
        self.contractions = {
            "aint": "ain't",
            "arent": "aren't",
            "cant": "can't",
            "couldve": "could've",
            "couldnt": "couldn't",
            "couldn'tve": "couldn't've",
            "couldnt've": "couldn't've",
            "didnt": "didn't",
            "doesnt": "doesn't",
            "dont": "don't",
            "hadnt": "hadn't",
            "hadnt've": "hadn't've",
            "hadn'tve": "hadn't've",
            "hasnt": "hasn't",
            "havent": "haven't",
            "hed": "he'd",
            "hed've": "he'd've",
            "he'dve": "he'd've",
            "hes": "he's",
            "howd": "how'd",
            "howll": "how'll",
            "hows": "how's",
            "Id've": "I'd've",
            "I'dve": "I'd've",
            "Im": "I'm",
            "Ive": "I've",
            "isnt": "isn't",
            "itd": "it'd",
            "itd've": "it'd've",
            "it'dve": "it'd've",
            "itll": "it'll",
            "let's": "let's",
            "maam": "ma'am",
            "mightnt": "mightn't",
            "mightnt've": "mightn't've",
            "mightn'tve": "mightn't've",
            "mightve": "might've",
            "mustnt": "mustn't",
            "mustve": "must've",
            "neednt": "needn't",
            "notve": "not've",
            "oclock": "o'clock",
            "oughtnt": "oughtn't",
            "ow's'at": "'ow's'at",
            "'ows'at": "'ow's'at",
            "'ow'sat": "'ow's'at",
            "shant": "shan't",
            "shed've": "she'd've",
            "she'dve": "she'd've",
            "she's": "she's",
            "shouldve": "should've",
            "shouldnt": "shouldn't",
            "shouldnt've": "shouldn't've",
            "shouldn'tve": "shouldn't've",
            "somebody'd": "somebodyd",
            "somebodyd've": "somebody'd've",
            "somebody'dve": "somebody'd've",
            "somebodyll": "somebody'll",
            "somebodys": "somebody's",
            "someoned": "someone'd",
            "someoned've": "someone'd've",
            "someone'dve": "someone'd've",
            "someonell": "someone'll",
            "someones": "someone's",
            "somethingd": "something'd",
            "somethingd've": "something'd've",
            "something'dve": "something'd've",
            "somethingll": "something'll",
            "thats": "that's",
            "thered": "there'd",
            "thered've": "there'd've",
            "there'dve": "there'd've",
            "therere": "there're",
            "theres": "there's",
            "theyd": "they'd",
            "theyd've": "they'd've",
            "they'dve": "they'd've",
            "theyll": "they'll",
            "theyre": "they're",
            "theyve": "they've",
            "twas": "'twas",
            "wasnt": "wasn't",
            "wed've": "we'd've",
            "we'dve": "we'd've",
            "weve": "we've",
            "werent": "weren't",
            "whatll": "what'll",
            "whatre": "what're",
            "whats": "what's",
            "whatve": "what've",
            "whens": "when's",
            "whered": "where'd",
            "wheres": "where's",
            "whereve": "where've",
            "whod": "who'd",
            "whod've": "who'd've",
            "who'dve": "who'd've",
            "wholl": "who'll",
            "whos": "who's",
            "whove": "who've",
            "whyll": "why'll",
            "whyre": "why're",
            "whys": "why's",
            "wont": "won't",
            "wouldve": "would've",
            "wouldnt": "wouldn't",
            "wouldnt've": "wouldn't've",
            "wouldn'tve": "wouldn't've",
            "yall": "y'all",
            "yall'll": "y'all'll",
            "y'allll": "y'all'll",
            "yall'd've": "y'all'd've",
            "y'alld've": "y'all'd've",
            "y'all'dve": "y'all'd've",
            "youd": "you'd",
            "youd've": "you'd've",
            "you'dve": "you'd've",
            "youll": "you'll",
            "youre": "you're",
            "youve": "you've",
        }
        self.manualMap = {
            "none": "0",
            "zero": "0",
            "one": "1",
            "two": "2",
            "three": "3",
            "four": "4",
            "five": "5",
            "six": "6",
            "seven": "7",
            "eight": "8",
            "nine": "9",
            "ten": "10",
        }
        self.articles = ["a", "an", "the"]

        self.periodStrip = re.compile("(?!<=\d)(\.)(?!\d)")
        self.commaStrip = re.compile("(\d)(,)(\d)")
        self.punct = [
            ";",
            r"/",
            "[",
            "]",
            '"',
            "{",
            "}",
            "(",
            ")",
            "=",
            "+",
            "\\",
            "_",
            "-",
            ">",
            "<",
            "@",
            "`",
            ",",
            "?",
            "!",
        ]

    def evaluate(self, quesIds=None):
        if quesIds is None:
            quesIds = [quesId for quesId in self.params["question_id"]]
        gts = {}
        res = {}
        for quesId in quesIds:
            gts[quesId] = self.vqa.qa[quesId]
            res[quesId] = self.vqaRes.qa[quesId]

        # =================================================
        # Compute accuracy
        # =================================================
        accQA = []
        accQuesType = {}
        accAnsType = {}
        print("computing accuracy")
        step = 0
        for quesId in quesIds:
            resAns = res[quesId]["answer"]
            resAns = resAns.replace("\n", " ")
            resAns = resAns.replace("\t", " ")
            resAns = resAns.strip()
            resAns = self.processPunctuation(resAns)
            resAns = self.processDigitArticle(resAns)
            gtAcc = []
            gtAnswers = [ans["answer"] for ans in gts[quesId]["answers"]]
            if len(set(gtAnswers)) > 1:
                for ansDic in gts[quesId]["answers"]:
                    ansDic["answer"] = self.processPunctuation(ansDic["answer"])
            for gtAnsDatum in gts[quesId]["answers"]:
                otherGTAns = [item for item in gts[quesId]["answers"] if item != gtAnsDatum]
                matchingAns = [item for item in otherGTAns if item["answer"] == resAns]
                acc = min(1, float(len(matchingAns)) / 3)
                gtAcc.append(acc)
            quesType = gts[quesId]["question_type"]
            ansType = gts[quesId]["answer_type"]
            avgGTAcc = float(sum(gtAcc)) / len(gtAcc)
            accQA.append(avgGTAcc)
            if quesType not in accQuesType:
                accQuesType[quesType] = []
            accQuesType[quesType].append(avgGTAcc)
            if ansType not in accAnsType:
                accAnsType[ansType] = []
            accAnsType[ansType].append(avgGTAcc)
            self.setEvalQA(quesId, avgGTAcc)
            self.setEvalQuesType(quesId, quesType, avgGTAcc)
            self.setEvalAnsType(quesId, ansType, avgGTAcc)
            if step % 100 == 0:
                self.updateProgress(step / float(len(quesIds)))
            step = step + 1

        self.setAccuracy(accQA, accQuesType, accAnsType)
        print("Done computing accuracy")

    def processPunctuation(self, inText):
        outText = inText
        for p in self.punct:
            if (p + " " in inText or " " + p in inText) or (re.search(self.commaStrip, inText) is not None):
                outText = outText.replace(p, "")
            else:
                outText = outText.replace(p, " ")
        outText = self.periodStrip.sub("", outText, re.UNICODE)
        return outText

    def processDigitArticle(self, inText):
        outText = []
        tempText = inText.lower().split()
        for word in tempText:
            word = self.manualMap.setdefault(word, word)
            if word not in self.articles:
                outText.append(word)
            else:
                pass
        for wordId, word in enumerate(outText):
            if word in self.contractions:
                outText[wordId] = self.contractions[word]
        outText = " ".join(outText)
        return outText

    def setAccuracy(self, accQA, accQuesType, accAnsType):
        self.accuracy["overall"] = round(100 * float(sum(accQA)) / len(accQA), self.n)
        self.accuracy["perQuestionType"] = {
            quesType: round(
                100 * float(sum(accQuesType[quesType])) / len(accQuesType[quesType]),
                self.n,
            )
            for quesType in accQuesType
        }
        self.accuracy["perAnswerType"] = {
            ansType: round(100 * float(sum(accAnsType[ansType])) / len(accAnsType[ansType]), self.n)
            for ansType in accAnsType
        }

    def setEvalQA(self, quesId, acc):
        self.evalQA[quesId] = round(100 * acc, self.n)

    def setEvalQuesType(self, quesId, quesType, acc):
        if quesType not in self.evalQuesType:
            self.evalQuesType[quesType] = {}
        self.evalQuesType[quesType][quesId] = round(100 * acc, self.n)

    def setEvalAnsType(self, quesId, ansType, acc):
        if ansType not in self.evalAnsType:
            self.evalAnsType[ansType] = {}
        self.evalAnsType[ansType][quesId] = round(100 * acc, self.n)

    def updateProgress(self, progress):
        barLength = 20
        status = ""
        if isinstance(progress, int):
            progress = float(progress)
        if not isinstance(progress, float):
            progress = 0
            status = "error: progress var must be float\r\n"
        if progress < 0:
            progress = 0
            status = "Halt...\r\n"
        if progress >= 1:
            progress = 1
            status = "Done...\r\n"
        block = int(round(barLength * progress))
        text = "\rFinshed Percent: [{0}] {1}% {2}".format(
            "#" * block + "-" * (barLength - block), int(progress * 100), status
        )
        sys.stdout.write(text)
        sys.stdout.flush()


class VQA:
    def __init__(self, annotation_file=None, question_file=None):
        """
        Constructor of VQA helper class for reading and visualizing questions and answers.
        :param annotation_file (str): location of VQA annotation file
        :return:
        """
        # load dataset
        self.dataset = {}
        self.questions = {}
        self.qa = {}
        self.qqa = {}
        self.imgToQA = {}
        if annotation_file is not None and question_file is not None:
            print("loading VQA annotations and questions into memory...")
            # time_t = datetime.datetime.utcnow()
            dataset = json.load(open(annotation_file, "r"))
            questions = json.load(open(question_file, "r"))
            self.dataset = dataset
            self.questions = questions
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        imgToQA = {ann["image_id"]: [] for ann in self.dataset["annotations"]}
        qa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        qqa = {ann["question_id"]: [] for ann in self.dataset["annotations"]}
        for ann in self.dataset["annotations"]:
            imgToQA[ann["image_id"]] += [ann]
            qa[ann["question_id"]] = ann
        for ques in self.questions["questions"]:
            qqa[ques["question_id"]] = ques
        print("index created!")

        # create class members
        self.qa = qa
        self.qqa = qqa
        self.imgToQA = imgToQA

    def info(self):
        """
        Print information about the VQA annotation file.
        :return:
        """
        for key, value in self.datset["info"].items():
            print("%s: %s" % (key, value))

    def getQuesIds(self, imgIds=[], quesTypes=[], ansTypes=[]):
        """
        Get question ids that satisfy given filter conditions. default skips that filter
        :param 	imgIds    (int array)   : get question ids for given imgs
                        quesTypes (str array)   : get question ids for given question types
                        ansTypes  (str array)   : get question ids for given answer types
        :return:    ids   (int array)   : integer array of question ids
        """
        imgIds = imgIds if type(imgIds) == list else [imgIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(imgIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                anns = sum(
                    [self.imgToQA[imgId] for imgId in imgIds if imgId in self.imgToQA],
                    [],
                )
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["question_id"] for ann in anns]
        return ids

    def getImgIds(self, quesIds=[], quesTypes=[], ansTypes=[]):
        """
         Get image ids that satisfy given filter conditions. default skips that filter
         :param quesIds   (int array)   : get image ids for given question ids
        quesTypes (str array)   : get image ids for given question types
        ansTypes  (str array)   : get image ids for given answer types
         :return: ids     (int array)   : integer array of image ids
        """
        quesIds = quesIds if type(quesIds) == list else [quesIds]
        quesTypes = quesTypes if type(quesTypes) == list else [quesTypes]
        ansTypes = ansTypes if type(ansTypes) == list else [ansTypes]

        if len(quesIds) == len(quesTypes) == len(ansTypes) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(quesIds) == 0:
                anns = sum([self.qa[quesId] for quesId in quesIds if quesId in self.qa], [])
            else:
                anns = self.dataset["annotations"]
            anns = anns if len(quesTypes) == 0 else [ann for ann in anns if ann["question_type"] in quesTypes]
            anns = anns if len(ansTypes) == 0 else [ann for ann in anns if ann["answer_type"] in ansTypes]
        ids = [ann["image_id"] for ann in anns]
        return ids

    def loadQA(self, ids=[]):
        """
        Load questions and answers with the specified question ids.
        :param ids (int array)       : integer ids specifying question ids
        :return: qa (object array)   : loaded qa objects
        """
        if type(ids) == list:
            return [self.qa[id] for id in ids]
        elif type(ids) == int:
            return [self.qa[ids]]

    def showQA(self, anns):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        for ann in anns:
            quesId = ann["question_id"]
            print("Question: %s" % (self.qqa[quesId]["question"]))
            for ans in ann["answers"]:
                print("Answer %d: %s" % (ans["answer_id"], ans["answer"]))

    def loadRes(self, resFile, quesFile):
        """
        Load result file and return a result object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = VQA()
        res.questions = json.load(open(quesFile))
        res.dataset["info"] = copy.deepcopy(self.questions["info"])
        res.dataset["task_type"] = copy.deepcopy(self.questions["task_type"])
        res.dataset["data_type"] = copy.deepcopy(self.questions["data_type"])
        res.dataset["data_subtype"] = copy.deepcopy(self.questions["data_subtype"])
        res.dataset["license"] = copy.deepcopy(self.questions["license"])

        print("Loading and preparing results...     ")
        time_t = datetime.datetime.utcnow()
        anns = json.load(open(resFile))
        assert type(anns) == list, "results is not an array of objects"
        annsQuesIds = [ann["question_id"] for ann in anns]
        assert set(annsQuesIds) == set(
            self.getQuesIds()
        ), "Results do not correspond to current VQA set. Either the results do not have predictions for all question ids in annotation file or there is atleast one question id that does not belong to the question ids in the annotation file."
        for ann in anns:
            quesId = ann["question_id"]
            if res.dataset["task_type"] == "Multiple Choice":
                assert (
                    ann["answer"] in self.qqa[quesId]["multiple_choices"]
                ), "predicted answer is not one of the multiple choices"
            qaAnn = self.qa[quesId]
            ann["image_id"] = qaAnn["image_id"]
            ann["question_type"] = qaAnn["question_type"]
            ann["answer_type"] = qaAnn["answer_type"]
        print("DONE (t=%0.2fs)" % ((datetime.datetime.utcnow() - time_t).total_seconds()))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res
