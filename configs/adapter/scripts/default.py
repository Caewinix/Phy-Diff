def adapter_condition_getter(config):
    def wrapper(data, device):
        return (data[-1].to(device),)
    return wrapper