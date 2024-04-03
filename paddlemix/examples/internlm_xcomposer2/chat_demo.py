import argparse
import paddle

from paddlemix.models.internlm_xcomposer2.modeling import InternLMXComposer2ForCausalLM
from paddlemix.models.internlm_xcomposer2.tokenization import InternLMXComposer2Tokenizer

paddle.set_grad_enabled(False)

parser = argparse.ArgumentParser()
# parser.add_argument("--from_pretrained", type=str, default="internlm/internlm-xcomposer2-7b", help="pretrained ckpt and tokenizer")
parser.add_argument("--from_pretrained", type=str, default="/home/ma-user/work/yk/pd/pd_internlm/internlm_xcomposer2_7b", help="pretrained ckpt and tokenizer")
args = parser.parse_args()

MODEL_PATH = args.from_pretrained
# init model and tokenizer
model = InternLMXComposer2ForCausalLM.from_pretrained(MODEL_PATH).eval()
tokenizer = InternLMXComposer2Tokenizer.from_pretrained(MODEL_PATH)

text = '<ImageHere>Please describe this image in detail.'
image = '../image1.jpg'
with paddle.no_grad():
    response, _ = model.chat(tokenizer, query=text, image=image, history=[], do_sample=False)
print(response)
