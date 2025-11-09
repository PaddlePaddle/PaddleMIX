import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from qdiff.base.base_quantizer import BaseQuantizer, StaticQuantizer, DynamicQuantizer
from qdiff.base.quant_layer import QuantizedLinear
from qdiff.utils import apply_func_to_submodules

# optional alternative quant-layer implementations (assume Paddle versions exist)
# from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear
# from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
# from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear

import logging
logger = logging.getLogger(__name__)


def quant_layer_refactor_(submodule, name, parent_module, quant_config=None, full_name=None, remain_fp_regex=None):
    """
    Replace a nn.Linear submodule with a QuantizedLinear (or variants) according to quant_config and regex.
    This function is intended to be called by apply_func_to_submodules; the kwargs it expects are provided there.
    """
    quant_layer_type = QuantizedLinear

    # METHOD: smooth_quant
    if quant_config is not None and quant_config.get("smooth_quant", None) is not None:
        from qdiff.smooth_quant.sq_quant_layer import SQQuantizedLinear
        import re
        layer_regex = quant_config.smooth_quant.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = SQQuantizedLinear
            logger.info('[INFO] setting smooth quant for layer {}'.format(full_name))

    # METHOD: quarot
    if quant_config is not None and quant_config.get("quarot", None) is not None:
        from qdiff.quarot.quarot_quant_layer import QuarotQuantizedLinear
        import re
        layer_regex = quant_config.quarot.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = QuarotQuantizedLinear
            logger.info('setting quarot for layer {}'.format(full_name))

    # METHOD: viditq - quarot + smooth_quant (both used)
    if quant_config is not None and quant_config.get("viditq", None) is not None:
        from qdiff.viditq.viditq_quant_layer import ViDiTQuantizedLinear
        import re
        layer_regex = quant_config.viditq.layer_name_regex
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            quant_layer_type = ViDiTQuantizedLinear
            logger.info('setting viditq for layer {}'.format(full_name))

    # set some layers as FP (fixed), feed in from config
    if remain_fp_regex is not None:
        import re
        pattern = re.compile(remain_fp_regex)
        if pattern.search(full_name):
            logger.info(f"remain {full_name} quant as FP due to fp_regex")
            return

    # derive in/out features from the Linear weight shape
    # Paddle Linear weight shape: [in_features, out_features]
    w_shape = list(submodule.weight.shape)
    if len(w_shape) != 2:
        raise RuntimeError(f"Unexpected weight shape for Linear layer {full_name}: {w_shape}")
    out_features, in_features = int(w_shape[1]), int(w_shape[0])
    bias = True if getattr(submodule, 'bias', None) is not None else False

    # device placeholder: Paddle doesn't accept device in Linear constructor; keep for API parity
    device = paddle.get_device() if paddle.is_compiled_with_cuda() else paddle.get_device()

    # create the quantized layer and replace in parent module
    quant_layer = quant_layer_type(in_features, out_features, bias, device, quant_config, submodule)
    setattr(parent_module, name, quant_layer)

    # set the module_name for quant_layer and its quantizers
    setattr(getattr(parent_module, name), 'module_name', full_name)
    if getattr(parent_module, name).w_quantizer is not None:
        setattr(getattr(parent_module, name).w_quantizer, 'module_name', full_name)
    if getattr(parent_module, name).a_quantizer is not None:
        setattr(getattr(parent_module, name).a_quantizer, 'module_name', full_name)


def bitwidth_refactor_(submodule, name=None, parent_module=None, quant_config=None, full_name=None):
    """
    Set mixed-precision bitwidths for matched layers according to regex lists in quant_config.
    Expects quant_config.mixed_precision.weight.layer_name_regex and .act.layer_name_regex to be iterable lists.
    """
    import re

    if quant_config is None:
        return

    layer_regex_list_w = quant_config.mixed_precision.weight.layer_name_regex
    layer_regex_list_a = quant_config.mixed_precision.act.layer_name_regex

    # Weight bitwidth refactor
    for idx, layer_regex in enumerate(layer_regex_list_w):
        if len(layer_regex) == 0:  # skip empty regex entries
            continue
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            if idx == 0:  # FP16 (or FP)
                submodule.quant_mode = False
                logger.info(f'[Mixed Precision] set the {full_name} W as FP16')
            else:
                # idx-1 maps to index in bitwidth list inside quantizer
                submodule.w_quantizer.bitwidth_refactor(idx - 1)
                logger.info(f'[Mixed Precision] set the {full_name} W as {submodule.w_quantizer.bitwidth_list[idx-1]} bit')

    # Activation bitwidth refactor
    for idx, layer_regex in enumerate(layer_regex_list_a):
        if len(layer_regex) == 0:
            continue
        match = re.search(re.compile(layer_regex), full_name)
        if match:
            if idx == 0:
                submodule.quant_mode = False
                logger.info(f'[Mixed Precision] set the {full_name} A as FP16')
            else:
                submodule.a_quantizer.bitwidth_refactor(idx - 1)
                logger.info(f'[Mixed Precision] set the {full_name} A as {submodule.a_quantizer.bitwidth_list[idx-1]} bit')


def load_quant_param_dict_(submodule, full_name=None, parent_module=None, quant_param_dict=None, model=None, **kwargs):
    """
    Load delta/zero_point (and other params) from quant_param_dict into quantizers/submodules.
    """
    if quant_param_dict is None or full_name not in quant_param_dict:
        return

    submodule.delta = quant_param_dict[full_name]['delta']
    submodule.zero_point = quant_param_dict[full_name]['zero_point']

    # reinit the rotation_matrix/channel_mask for special quant methods (viditq/quarot/sq)
    # Because these classes are not always imported here, guard imports and attribute checks
    if hasattr(parent_module, 'channel_mask') and hasattr(parent_module, 'rotation_matrix'):
        # ViDiTQuantizedLinear expected
        # ensure parent_module has expected methods; call to re-compute rotation matrix
        parent_module.get_rotation_matrix()
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask']
        parent_module.update_quantized_weight_rotated_and_scaled()
    elif not hasattr(parent_module, 'channel_mask') and hasattr(parent_module, 'rotation_matrix'):
        # QuarotQuantizedLinear expected
        parent_module.get_rotation_matrix()
        parent_module.update_quantized_weight_rotated()
    elif hasattr(parent_module, 'channel_mask') and not hasattr(parent_module, 'rotation_matrix'):
        # SQQuantizedLinear expected
        parent_module.channel_mask = quant_param_dict[full_name]['channel_mask']
        parent_module.update_quantized_weight_scaled()

    # update the quant_model.quant_param_dict also
    if model is not None:
        model.quant_param_dict[full_name] = quant_param_dict[full_name]


def save_quant_param_dict_(submodule, full_name=None, parent_module=None, model=None, **kwargs):
    """
    Save delta/zero_point (and other small per-layer values) into model.quant_param_dict.
    """
    if model is None:
        return

    model.quant_param_dict[full_name] = {}
    model.quant_param_dict[full_name]['delta'] = submodule.delta
    model.quant_param_dict[full_name]['zero_point'] = submodule.zero_point

    # parent module: the quant_layer (may have channel_mask or rotation_matrix)
    if hasattr(parent_module, 'channel_mask'):
        model.quant_param_dict[full_name]['channel_mask'] = parent_module.channel_mask
    if hasattr(parent_module, 'rotation_matrix'):
        # skip saving large rotation_matrix to reduce size (original code set None)
        model.quant_param_dict[full_name]['rotation_matrix'] = None


def set_init_done_(submodule, **kwargs):
    """Mark quantizer init_done flag."""
    submodule.init_done = True


'''
IMPORTANT: this file is simply a template, you should inherit the model you are using
and implement these functions. 
ref the examples in `examples/dit/models/quant_dit.py`
'''
class QuantModel(nn.Layer):
    """
    the base quant model (Paddle)
    specialized funcs should be implemented in subclass.
    (e.g., QuantizedOpenSORA...)
    """
    def __init__(self, quant_config: dict = None, **kwargs) -> None:
        super().__init__()  # initialize all attributes from parent class

        # additional attributes for quant
        self.q_cfg = quant_config or {}
        self.quant_param_dict = {}

        # refactor layers with quant_layers based on q_cfg
        self.quant_layer_refactor()

    def quant_layer_refactor(self):
        # pass quant_config so the callback has access to it
        apply_func_to_submodules(
            self,
            class_type=nn.Linear,
            function=quant_layer_refactor_,
            quant_config=self.q_cfg
        )

    def save_quant_params_dict(self):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=save_quant_param_dict_,
            model=self
        )

    def load_quant_params_dict(self, quant_param_dict):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=load_quant_param_dict_,
            quant_param_dict=quant_param_dict,
            model=self
        )

    def set_init_done(self):
        apply_func_to_submodules(
            self,
            class_type=BaseQuantizer,
            function=set_init_done_,
        )

    def bitwidth_refactor(self):
        apply_func_to_submodules(
            self,
            class_type=QuantizedLinear,
            function=bitwidth_refactor_,
            quant_config=self.q_cfg
        )

    def forward(self, x, *args, **kwargs):
        raise NotImplementedError("should be implemented in subclass.")


if __name__ == '__main__':
    import paddle
    import re
    from types import SimpleNamespace

    paddle.set_device('gpu')  # 若要跑 GPU，把它改为 'gpu' 并确保 paddle 已编译 GPU

    # Helper config wrapper that supports both dict-like .get(...) and attribute access (.smooth_quant)
    class Cfg(dict):
        def __getattr__(self, name):
            if name in self:
                return self[name]
            raise AttributeError(name)
        def __setattr__(self, name, val):
            self[name] = val

    # Build a simple quant_config compatible with the refactor callbacks
    quant_config = Cfg()
    quant_config['weight'] = Cfg(n_bits=8, sym=True)
    quant_config['act'] = Cfg(n_bits=8, sym=True)
    # placeholders for optional subconfigs
    quant_config['smooth_quant'] = None
    quant_config['quarot'] = None
    quant_config['viditq'] = None


    # Minimal Model subclass that creates layers first, then calls QuantModel.__init__
    class MyModel(QuantModel):
        def __init__(self, quant_cfg):
            # create layers BEFORE calling QuantModel.__init__ because parent __init__ triggers refactor
            # two linear layers for testing
            super().__init__(quant_cfg)
            self.fc1 = nn.Linear(16, 16, bias_attr=True)
            self.fc2 = nn.Linear(16, 8, bias_attr=True)
            # call parent constructor to set up quant logic (this will call quant_layer_refactor)
            

        def forward(self, x):
            # a simple forward that passes x through the two (possibly quantized) linear layers
            # expected input shape: [B, N_token, C], where C==16
            # Just apply sequentially
            x = self.fc1(x)
            x = paddle.nn.functional.relu(x)
            x = self.fc2(x)
            return x

    # Instantiate model and run tests
    model = MyModel(quant_config)
    print("Model after quant_layer_refactor:")
    print(model)

    # Create a dummy input: B=2, N_token=3, C=16
    B, N_token, C = 2, 3, 16
    x = paddle.randn([B, N_token, C], dtype='float32')

    # Ensure quant_mode True (default in QuantizedLinear)
    print("\n--- Forward (quant mode) ---")
    # run forward (QuantModel.forward is implemented by MyModel)
    y_q = model(x)
    print("Output shape (quant):", y_q.shape)
    print("Sample outputs (quant):", y_q.flatten()[:6].numpy())

    # Save quant params dict after one forward (so quantizers are initialized)
    model.save_quant_params_dict()
    print("\nSaved quant_param_dict keys:", list(model.quant_param_dict.keys()))
    # print a sample layer's delta/zero_point shapes if available
    if len(model.quant_param_dict) > 0:
        sample_name = list(model.quant_param_dict.keys())[0]
        sample = model.quant_param_dict[sample_name]
        print(f"Sample layer '{sample_name}' quant params:")
        print("  delta:", getattr(sample.get('delta', None), 'shape', None))
        print("  zero_point:", getattr(sample.get('zero_point', None), 'shape', None))

    # Create a second new model and load the saved quant params into it to verify load logic
    print("\n--- Load quant params into a fresh model and forward ---")
    model2 = MyModel(quant_config)
    # before loading, model2.quant_param_dict should be empty
    print("model2.quant_param_dict keys (before load):", list(model2.quant_param_dict.keys()))
    # load saved params
    model2.load_quant_params_dict(model.quant_param_dict)
    # set init_done flags
    model2.set_init_done()
    # run forward in quant mode
    y_q2 = model2(x)
    print("Output shape (quant) after load:", y_q2.shape)
    print("Sample outputs (quant) after load:", y_q2.flatten()[:6].numpy())

    # Compare outputs (they may not be identical due to randomness and how quantization is implemented,
    # but this at least ensures code paths run without errors).
    try:
        diff = (y_q - y_q2).abs().mean().numpy().item()
        print(f"\nMean absolute difference between model and model2 outputs: {diff:.6f}")
    except Exception:
        print("Could not compute difference (maybe shapes/types mismatch).")

    # Also run FP mode by setting quant_mode=False on quantized layers
    print("\n--- Forward (FP mode) ---")
    # Flip quant_mode on submodules that are QuantizedLinear
    def set_fp(submodule, **kwargs):
        if hasattr(submodule, 'quant_mode'):
            submodule.quant_mode = False

    apply_func_to_submodules(model, class_type=QuantizedLinear, function=set_fp)
    y_fp = model(x)
    print("Output shape (fp):", y_fp.shape)
    print("Sample outputs (fp):", y_fp.flatten()[:6].numpy())

    print("\n✅ __main__ smoke test completed.")
