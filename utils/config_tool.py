import yaml
import argparse
from .mirror import main_globals

def parse_args(args: list):
    for i, unparsed_arg in enumerate(args):
        if isinstance(unparsed_arg, str):
            try:
                args[i] = eval(unparsed_arg, main_globals())
            except (SyntaxError, NameError) as _:
                pass
    return args

def parse_kwargs(kwargs: dict):
    for key, unparsed_kwarg in kwargs.items():
        if isinstance(unparsed_kwarg, str):
            try:
                kwargs[key] = eval(unparsed_kwarg, main_globals())
            except (SyntaxError, NameError) as _:
                pass
    return kwargs

class script_func:
    def __init__(self, func, args):
        self.__func = func
        self.__args = args
        self.__has_parsed = False
    
    def __call__(self, *args):
        return self.__func(*args)
    
    def call(self, *args):
        return self(*args, *self.args)
    
    @property
    def args(self):
        if not self.__has_parsed:
            self.__has_parsed = True
            parse_args(self.__args)
        return self.__args

class NullableNamespace(argparse.Namespace):
    def __getattr__(self, name: str):
        return None
    
def dict_to_namespace(config):
    root_namespace = NullableNamespace()
    stack = [(root_namespace, config)]
    while stack:
        namespace, current_dict = stack.pop()
        for key, value in current_dict.items():
            key = key.replace('-', '_')
            if isinstance(value, dict):
                if ('script' in value) and value['script']:
                    if 'name' in value:
                        script_name = value['name']
                    else:
                        script_name = key
                    exec(f"from {config['script']} import {script_name}")
                    if 'args' in value:
                        setattr(namespace, key, script_func(eval(script_name), value['args']))
                    else:
                        setattr(namespace, key, eval(script_name))
                else:
                    new_namespace = NullableNamespace()
                    setattr(namespace, key, new_namespace)
                    stack.append((new_namespace, value))
            else:
                setattr(namespace, key, value)
    return root_namespace

def raw_dict_to_namespace(config):
    root_namespace = NullableNamespace()
    stack = [(root_namespace, config)]
    while stack:
        namespace, current_dict = stack.pop()
        for key, value in current_dict.items():
            if isinstance(value, dict):
                new_namespace = NullableNamespace()
                setattr(namespace, key, new_namespace)
                stack.append((new_namespace, value))
            else:
                setattr(namespace, key, value)
    return root_namespace

def load(path: str):
    with open(path, 'r') as file:
        config = yaml.safe_load(file)
        config = dict_to_namespace(config)
    return config
    