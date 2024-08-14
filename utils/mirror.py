import dis
import types
from collections.abc import Iterator
import inspect
from opcode import HAVE_ARGUMENT

class InstructionOperationIterator(Iterator):
    def __init__(self, iterable):
        self.__iterator = iterable.__iter__()
    def __iter__(self):
        return self
    def __next__(self):
        instructions = self.__iterator.__next__()
        return instructions.opcode, instructions.arg

def instructions_to_bytecode(instructions: list[dis.Instruction]):
    return reverse_opargs(InstructionOperationIterator(instructions))

def reverse_opargs(op_arg_arr):
    code = bytearray()
    for op, arg in op_arg_arr:
        if op < HAVE_ARGUMENT:
            # For opcodes without an argument, add only the op
            arg = 0
        else:
            # For opcodes with an argument, handle potential extended arguments
            while arg > 255:
                arg = arg & 0xff  # Keep only the lower byte for the final argument
        code.extend([op, arg])
    return bytes(code)

def _find_super_init_pairs(bytecode):
    pairs = []
    init_index = None  # Track the index of the last seen __init__
    for i, instr in enumerate(bytecode):
        if instr.argval == '__init__' and bytecode[i - 2].argval == 'super':
            init_index = i # Found an __init__, remember its index
        # Check if this is a POP_TOP and we have seen an __init__ before it
        if instr.opname == 'POP_TOP' and init_index is not None:
            # Ensure the POP_TOP is after the __init__
            if i > init_index:
                pairs.append((init_index - 2, i + 1))
                init_index = None  # Reset init_index for the next pair
    return pairs

def modify_func(func, instructions: list[dis.Instruction], func_name: str = ''):
    code = types.CodeType(
        func.__code__.co_argcount,
        func.__code__.co_posonlyargcount,
        func.__code__.co_kwonlyargcount,
        func.__code__.co_nlocals,
        func.__code__.co_stacksize,
        func.__code__.co_flags,
        instructions_to_bytecode(instructions),
        func.__code__.co_consts,
        func.__code__.co_names,
        func.__code__.co_varnames,
        func.__code__.co_filename,
        func_name,
        func.__code__.co_firstlineno,
        func.__code__.co_lnotab
    )
    return types.FunctionType(code, globals())

def main_globals():
    stack = inspect.stack()
    for frame_info in stack:
        if frame_info.frame.f_globals.get('__name__') == '__main__':
            main_globals = frame_info.frame.f_globals
            break
    return main_globals