import inspect
from typing import Iterable

class ArgsContext:
    def __init__(self, **kwargs):
        self.__args_context = kwargs
    
    def __getitem__(self, name):
        return self.__args_context[name]
    
    def __getattr__(self, name):
        return self.__args_context[name]
    
    @classmethod
    def __class_getitem__(cls, typename):
        if isinstance(typename, tuple):
            typename, exception_types = typename
            if not isinstance(exception_types, Iterable):
                exception_types = [exception_types]
        else:
            exception_types = []
        class ArgsContext:
            def __init__(self, *args, **kwargs):
                signature = inspect.signature(typename.__init__)
                parameters = signature.parameters
                parameters = [p for p in signature.parameters.values() if p.name != 'self']
                signature = signature.replace(parameters=parameters)
                bound_arguments = signature.bind(*args, **kwargs)
                bound_arguments.apply_defaults()
                for name, value in bound_arguments.arguments.items():
                    if not name in exception_types:
                        setattr(self, name, value)
        return ArgsContext
    
# import inspect
# from typing import Iterable

# class ArgsContext:
#     @classmethod
#     def __class_getitem__(cls, identifier):
#         if isinstance(identifier, tuple):
#             identifier, exception_types = identifier
#             if not isinstance(exception_types, Iterable):
#                 exception_types = [exception_types]
#         else:
#             exception_types = []
#         if inspect.isfunction(identifier):
#             inspected_func = identifier
#             def get_parameters_values(parameters_values):
#                 return list(parameters_values)
#         else:
#             if inspect.isclass(identifier):
#                 inspected_func = identifier.__init__
#             elif inspect.ismethod(identifier):
#                 inspected_func = identifier
#             def get_parameters_values(parameters_values):
#                 return [p for p in parameters_values() if p.name != 'self']
#         class ArgsContext:
#             def __init__(self, *args, **kwargs):
#                 signature = inspect.signature(inspected_func)
#                 parameters = signature.parameters
#                 # parameters = [p for p in signature.parameters.values() if p.name != 'self']
#                 parameters = get_parameters_values(parameters.values())
#                 signature = signature.replace(parameters=parameters)
#                 bound_arguments = signature.bind(*args, **kwargs)
#                 bound_arguments.apply_defaults()
#                 for name, value in bound_arguments.arguments.items():
#                     if not name in exception_types:
#                         setattr(self, name, value)
#         return ArgsContext