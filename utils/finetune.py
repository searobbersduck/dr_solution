

def copy_model_parameters(model, source_parameters, n_params=1e10):
    param_dict = {}
    if isinstance(source_parameters, list):
        for i, (key, value) in enumerate(model.state_dict().items()):
            if i > len(source_parameters) or i >= n_params:
                param_dict[key] = value
            else:
                assert source_parameters[i].numel() == model.state_dict()[key].numel()
                param_dict[key] = source_parameters[i].view(*value.size()).contiguous()
    elif isinstance(source_parameters, dict):
        for key,value in model.state_dict().items():
            if key not in source_parameters:
                param_dict[key] = value
            else:
                assert source_parameters[key].numel() == model.state_dict()[key].numel()
                param_dict[key] = source_parameters[key].view(*value.size()).contiguous()