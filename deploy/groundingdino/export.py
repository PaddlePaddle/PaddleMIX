import argparse
import os
import paddle
from paddle.static import InputSpec

from paddlemix.models.groundingdino.modeling import GroundingDinoModel


def _prune_input_spec(input_spec, program, targets):
    # try to prune static program to figure out pruned input spec
    # so we perform following operations in static mode

    device = paddle.get_device()
    paddle.enable_static()
    paddle.set_device(device)
    pruned_input_spec = [{}]
    program = program.clone()
    program = program._prune(targets=targets)
    global_block = program.global_block()

    for spec in input_spec:
        try:
            name = spec.name
            v = global_block.var(name)
            pruned_input_spec[0][name] = spec
        except Exception:
            pass
    paddle.disable_static(place=device)
    return pruned_input_spec


def apply_to_static(model):

    input_spec = [
        InputSpec(
            shape=[None, 3, None, None], name='x', dtype='float32'), InputSpec(
                shape=[None, None, None], name='m', dtype="int64"), InputSpec(
                    shape=[None, None], name='input_ids',
                    dtype="int64"), InputSpec(
                        shape=[None, None],
                        name='attention_mask',
                        dtype="int64"), InputSpec(
                            shape=[None, None, None],
                            name='text_self_attention_masks',
                            dtype="int64"), InputSpec(
                                shape=[None, None],
                                name='position_ids',
                                dtype="int64")
    ]
    model = paddle.jit.to_static(model, input_spec=input_spec)
    return model, input_spec


if __name__ == "__main__":

    parser = argparse.ArgumentParser("Grounding DINO example", add_help=True)
    parser.add_argument(
        "--dino_type",
        "-dt",
        type=str,
        default="GroundingDino/groundingdino-swint-ogc",
        help="dino type")
    parser.add_argument(
        "--output_dir",
        "-o",
        type=str,
        default="output_groundingdino",
        help="output directory")
    args = parser.parse_args()

    output_dir = args.output_dir
    # load model
    model = GroundingDinoModel.from_pretrained(args.dino_type)
    model.eval()

    static_model, input_spec = apply_to_static(model)

    paddle.jit.save(
        static_model,
        os.path.join(output_dir, 'groundingdino_model'),
        input_spec=input_spec)
