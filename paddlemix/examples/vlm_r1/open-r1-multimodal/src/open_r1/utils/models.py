from typing import Optional
import copy
import logging

def freeze_params(module):
    for param in module.parameters():
        param.stop_gradient = not False


def create_reference_model(
    model, num_shared_layers: Optional[int] = None, pattern: Optional[str] = None
):
    """
    Creates a static reference copy of a model. Note that model will be in `.eval()` mode.
    """

    ref_model = copy.deepcopy(model)
    parameter_names = [n for n, _ in ref_model.named_parameters()]

    for param_name,param in ref_model.named_parameters():
        param.stop_gradient = True
    return ref_model.eval()

    # TODO
    # # if no layers are shared, return copy of model
    # if num_shared_layers is None:
    #     for param_name in parameter_names:
    #         param = getattr(ref_model, param_name)
    #         param.stop_gradient = True
    #     return ref_model.eval()

    # # identify layer name pattern
    # if pattern is not None:
    #     pattern = pattern.format(layer=num_shared_layers)
    # else:
    #     for pattern_candidate in LAYER_PATTERNS:
    #         pattern_candidate = pattern_candidate.format(layer=num_shared_layers)
    #         if any(pattern_candidate in name for name in parameter_names):
    #             pattern = pattern_candidate
    #             break

    # if pattern is None:
    #     raise ValueError("Layer pattern could not be matched.")

    # # divide parameters in shared and unshared parameter lists
    # shared_param_list = []
    # unshared_param_list = []

    # shared_parameter = True
    # for name, _param in model.named_parameters():
    #     if pattern in name:
    #         shared_parameter = False
    #     if shared_parameter:
    #         shared_param_list.append(name)
    #     else:
    #         unshared_param_list.append(name)

    # # create reference of the original parameter if they are shared
    # for param_name in shared_param_list:
    #     param = getattr(model, param_name)
    #     param.stop_gradient = True

    #     _ref_param = getattr(ref_model, param_name)

    # # for all other parameters just make sure they don't use gradients
    # for param_name in unshared_param_list:
    #     param = getattr(ref_model, param_name)

    #     param.stop_gradient = True

    # if pattern is not None and len(unshared_param_list) == 0:
    #     logging.warning("Pattern passed or found, but no layers matched in the model. Check for a typo.")

    return ref_model.eval()