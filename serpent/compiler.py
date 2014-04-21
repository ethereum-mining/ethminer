#!/usr/bin/python
import re
import sys
import os
from parser import parse
from opcodes import opcodes, reverse_opcodes
import json

label_counter = [0]


def mklabel(prefix):
    label_counter[0] += 1
    return prefix + str(label_counter[0] - 1)

# All functions go here
#
# Entries go in a format:
#
# [ val, inputcount, outputcount, code ]

funtable = [
    ['+', 2, 1, ['<1>', '<0>', 'ADD']],
    ['-', 2, 1, ['<1>', '<0>', 'SUB']],
    ['*', 2, 1, ['<1>', '<0>', 'MUL']],
    ['/', 2, 1, ['<1>', '<0>', 'DIV']],
    ['^', 2, 1, ['<1>', '<0>', 'EXP']],
    ['%', 2, 1, ['<1>', '<0>', 'MOD']],
    ['#/', 2, 1, ['<1>', '<0>', 'SDIV']],
    ['#%', 2, 1, ['<1>', '<0>', 'SMOD']],
    ['==', 2, 1, ['<1>', '<0>', 'EQ']],
    ['<', 2, 1, ['<1>', '<0>', 'LT']],
    ['<=', 2, 1, ['<1>', '<0>', 'GT', 'NOT']],
    ['>', 2, 1, ['<1>', '<0>', 'GT']],
    ['>=', 2, 1, ['<1>', '<0>', 'LT', 'NOT']],
    ['!', 1, 1, ['<0>', 'NOT']],
    ['or', 2, 1, ['<1>', '<0>', 'DUP', 4, 'PC',
                  'ADD', 'JUMPI', 'POP', 'SWAP', 'POP']],
    ['||', 2, 1, ['<1>', '<0>', 'DUP', 4, 'PC',
                  'ADD', 'JUMPI', 'POP', 'SWAP', 'POP']],
    ['and', 2, 1, ['<1>', '<0>', 'NOT', 'NOT', 'MUL']],
    ['&&', 2, 1, ['<1>', '<0>', 'NOT', 'NOT', 'MUL']],
    ['xor', 2, 1, ['<1>', '<0>', 'XOR']],
    ['&', 2, 1, ['<1>', '<0>', 'AND']],
    ['|', 2, 1, ['<1>', '<0>', 'OR']],
    ['byte', 2, 1, ['<0>', '<1>', 'BYTE']],
    # Word array methods
    # arr, ind -> val
    ['access', 2, 1, ['<0>', '<1>', 32, 'MUL', 'ADD', 'MLOAD']],
    # arr, ind, val
    ['arrset', 3, 0, ['<2>', '<0>', '<1>', 32, 'MUL', 'ADD', 'MSTORE']],
    # val, pointer -> pointer+32
    ['set_and_inc', 2, 1, ['<1>', 'DUP', '<0>', 'SWAP', 'MSTORE', 32, 'ADD']],
    # len (32 MUL) len*32 (MSIZE) len*32 MSIZE (SWAP) MSIZE len*32 (MSIZE ADD)
    # MSIZE MSIZE+len*32 (1) MSIZE MSIZE+len*32 1 (SWAP SUB) MSIZE
    # MSIZE+len*32-1 (0 SWAP MSTORE8) MSIZE
    ['array', 1, 1, ['<0>', 32, 'MUL', 'MSIZE', 'SWAP', 'MSIZE',
                     'ADD', 1, 'SWAP', 'SUB', 0, 'SWAP', 'MSTORE8']],  # len -> arr
    # String array methods
    # arr, ind -> val
    ['getch', 2, 1, ['<1>', '<0>', 'ADD', 'MLOAD', 255, 'AND']],
    ['setch', 3, 0, ['<2>', '<1>', '<0>', 'ADD', 'MSTORE']],  # arr, ind, val
    # len MSIZE (SWAP) MSIZE len (MSIZE ADD) MSIZE MSIZE+len (1) MSIZE
    # MSIZE+len 1 (SWAP SUB) MSIZE MSIZE+len-1 (0 SWAP MSTORE8) MSIZE
    ['string', 1, 1, ['<0>', 'MSIZE', 'SWAP', 'MSIZE', 'ADD',
                      1, 'SWAP', 'SUB', 0, 'SWAP', 'MSTORE8']],  # len -> arr
    # ['send', 2, 1, [0,0,0,0,0,'<1>','<0>','CALL'] ], # to, value, 0, [] -> /dev/null
    # to, value, gas, [] -> /dev/null
    ['send', 3, 1, [0, 0, 0, 0, '<2>', '<1>', '<0>', 'CALL']],
    # MSIZE 0 MSIZE (MSTORE) MSIZE (DUP) MSIZE MSIZE (...) MSIZE MSIZE 32 <4>
    # <3> <2> <1> <0> (CALL) MSIZE FLAG (POP) MSIZE (MLOAD) RESULT
    ['msg', 5, 1, ['MSIZE', 0, 'MSIZE', 'MSTORE', 'DUP', 32, 'SWAP', '<4>', 32, 'MUL', '<3>',
                   '<2>', '<1>', '<0>', 'CALL', 'POP', 'MLOAD']],  # to, value, gas, data, datasize -> out32
    # <5>*32 (MSIZE SWAP MSIZE SWAP) MSIZE MSIZE <5>*32 (DUP MSIZE ADD) MSIZE MSIZE <5>*32 MEND+1 (1 SWAP SUB) MSIZE MSIZE <5>*32 MEND (0 SWAP MSTORE8) MSIZE MSIZE <5>*32 (SWAP) MSIZE <5>*32 MSIZE
    ['msg', 6, 1, ['<5>', 32, 'MUL', 'MSIZE', 'SWAP', 'MSIZE', 'SWAP', 'DUP', 'MSIZE', 'ADD', 1, 'SWAP', 'SUB', 0, 'SWAP', 'MSTORE8', 'SWAP',
                   '<4>', 32, 'MUL', '<3>', '<2>', '<1>', '<0>', 'CALL', 'POP']],  # to, value, gas, data, datasize, outsize -> out
    # value, gas, data, datasize
    ['create', 4, 1, ['<3>', '<2>', '<1>', '<0>', 'CREATE']],
    ['sha3', 1, 1, [32, 'MSIZE', '<0>', 'MSIZE', 'MSTORE', 'SHA3']],
    ['sha3bytes', 1, 1, ['SHA3']],
    ['sload', 1, 1, ['<0>', 'SLOAD']],
    ['sstore', 2, 0, ['<1>', '<0>', 'SSTORE']],
    ['calldataload', 1, 1, ['<0>', 32, 'MUL', 'CALLDATALOAD']],
    ['id', 1, 1, ['<0>']],
    # 0 MSIZE (SWAP) MSIZE 0 (MSIZE) MSIZE 0 MSIZE (MSTORE) MSIZE (32 SWAP) 32
    # MSIZE
    # returns single value
    ['return', 1, 0, [
        '<0>', 'MSIZE', 'SWAP', 'MSIZE', 'MSTORE', 32, 'SWAP', 'RETURN']],
    ['return', 2, 0, ['<1>', 32, 'MUL', '<0>', 'RETURN']],
    ['suicide', 1, 0, ['<0>', 'SUICIDE']],
]

# Pseudo-variables representing opcodes
pseudovars = {
    'msg.datasize': [32, 'CALLDATASIZE', 'DIV'],
    'msg.sender': ['CALLER'],
    'msg.value': ['CALLVALUE'],
    'tx.gasprice': ['GASPRICE'],
    'tx.origin': ['ORIGIN'],
    'tx.gas': ['GAS'],
    'contract.balance': ['BALANCE'],
    'block.prevhash': ['PREVHASH'],
    'block.coinbase': ['COINBASE'],
    'block.timestamp': ['TIMESTAMP'],
    'block.number': ['NUMBER'],
    'block.difficulty': ['DIFFICULTY'],
    'block.gaslimit': ['GASLIMIT'],
}


# A set of methods for detecting raw values (numbers and strings) and
# converting them to integers
def frombytes(b):
    return 0 if len(b) == 0 else ord(b[-1]) + 256 * frombytes(b[:-1])


def fromhex(b):
    return 0 if len(b) == 0 else '0123456789abcdef'.find(b[-1]) + 16 * fromhex(b[:-1])


def is_numberlike(b):
    if isinstance(b, (str, unicode)):
        if re.match('^[0-9\-]*$', b):
            return True
        if b[0] in ["'", '"'] and b[-1] in ["'", '"'] and b[0] == b[-1]:
            return True
        if b[:2] == '0x':
            return True
    return False


def numberize(b):
    if b[0] in ["'", '"']:
        return frombytes(b[1:-1])
    elif b[:2] == '0x':
        return fromhex(b[2:])
    else:
        return int(b)


# Apply rewrite rules
def rewrite(ast):
    if isinstance(ast, (str, unicode)):
        return ast
    elif ast[0] == 'set':
        if ast[1][0] == 'access':
            if ast[1][1] == 'contract.storage':
                return ['sstore', rewrite(ast[1][2]), rewrite(ast[2])]
            else:
                return ['arrset', rewrite(ast[1][1]), rewrite(ast[1][2]), rewrite(ast[2])]
    elif ast[0] == 'access':
        if ast[1] == 'msg.data':
            return ['calldataload', rewrite(ast[2])]
        elif ast[1] == 'contract.storage':
            return ['sload', rewrite(ast[2])]
    elif ast[0] == 'array_lit':
        o = ['array', str(len(ast[1:]))]
        for a in ast[1:]:
            o = ['set_and_inc', rewrite(a), o]
        return ['-', o, str(len(ast[1:])*32)]
    elif ast[0] == 'return':
        if len(ast) == 2 and ast[1][0] == 'array_lit':
            return ['return', rewrite(ast[1]), str(len(ast[1][1:]))]
    return map(rewrite, ast)


# Main compiler code
def arity(ast):
    if isinstance(ast, (str, unicode)):
        return 1
    elif ast[0] == 'set':
        return 0
    elif ast[0] == 'if':
        return 0
    elif ast[0] == 'seq':
        return 1 if len(ast[1:]) and arity(ast[-1]) == 1 else 0
    else:
        for f in funtable:
            if ast[0] == f[0]:
                return f[2]


# Debugging
def print_wrapper(f):
    def wrapper(*args, **kwargs):
        print args[0]
        u = f(*args, **kwargs)
        print u
        return u
    return wrapper


# Right-hand-side expressions (ie. the normal kind)
#@print_wrapper
def compile_expr(ast, varhash, lc=[0]):
    # Stop keyword
    if ast == 'stop':
        return ['STOP']
    # Literals
    elif isinstance(ast, (str, unicode)):
        if is_numberlike(ast):
            return [numberize(ast)]
        elif ast in pseudovars:
            return pseudovars[ast]
        else:
            if ast not in varhash:
                varhash[ast] = len(varhash) * 32
            return [varhash[ast], 'MLOAD']
    # Set (specifically, variables)
    elif ast[0] == 'set':
        if not isinstance(ast[1], (str, unicode)):
            raise Exception("Cannot set the value of " + str(ast[1]))
        elif ast[1] in pseudovars:
            raise Exception("Cannot set a pseudovariable!")
        else:
            if ast[1] not in varhash:
                varhash[ast[1]] = len(varhash) * 32
            return compile_expr(ast[2], varhash, lc) + [varhash[ast[1]], 'MSTORE']
    # If and if/else statements
    elif ast[0] == 'if':
        f = compile_expr(ast[1], varhash, lc)
        g = compile_expr(ast[2], varhash, lc)
        h = compile_expr(ast[3], varhash, lc) if len(ast) > 3 else None
        label, ref = 'LABEL_' + str(lc[0]), 'REF_' + str(lc[0])
        lc[0] += 1
        label2, ref2 = 'LABEL_' + str(lc[0]), 'REF_' + str(lc[0])
        lc[0] += 1
        if h:
            return f + ['NOT', ref2, 'JUMPI'] + g + [ref, 'JUMP', label2] + h + [label]
        else:
            return f + ['NOT', ref, 'JUMPI'] + g + [label]
    # While loops
    elif ast[0] == 'while':
        f = compile_expr(ast[1], varhash, lc)
        g = compile_expr(ast[2], varhash, lc)
        beglab, begref = 'LABEL_' + str(lc[0]), 'REF_' + str(lc[0])
        endlab, endref = 'LABEL_' + str(lc[0] + 1), 'REF_' + str(lc[0] + 1)
        lc[0] += 2
        return [beglab] + f + ['NOT', endref, 'JUMPI'] + g + [begref, 'JUMP', endlab]
    # Seq
    elif ast[0] == 'seq':
        o = []
        for arg in ast[1:]:
            o.extend(compile_expr(arg, varhash, lc))
            if arity(arg) == 1 and arg != ast[-1]:
                o.append('POP')
        return o
    # Functions and operations
    for f in funtable:
        if ast[0] == f[0] and len(ast[1:]) == f[1]:
            # If arity of all args is 1
            if reduce(lambda x, y: x * arity(y), ast[1:], 1):
                iq = f[3][:]
                oq = []
                while len(iq):
                    tok = iq.pop(0)
                    if isinstance(tok, (str, unicode)) and tok[0] == '<' and tok[-1] == '>':
                        oq.extend(
                            compile_expr(ast[1 + int(tok[1:-1])], varhash, lc))
                    else:
                        oq.append(tok)
                return oq
            else:
                raise Exception(
                    "Arity of argument mismatches for %s: %s" % (f[0], ast))
    raise Exception("invalid op: " + ast[0])


# Stuff to add once to each program
def add_wrappers(c, varhash):
    if len(varhash) and 'MSIZE' in c:
        return [0, len(varhash) * 32 - 1, 'MSTORE8'] + c
    else:
        return c


# Optimizations
ops = {
    'ADD': lambda x, y: (x + y) % 2 ** 256,
    'MUL': lambda x, y: (x * y) % 2 ** 256,
    'SUB': lambda x, y: (x - y) % 2 ** 256,
    'DIV': lambda x, y: x / y,
    'EXP': lambda x, y: pow(x, y, 2 ** 256),
    'AND': lambda x, y: x & y,
    'OR': lambda x, y: x | y,
    'XOR': lambda x, y: x ^ y
}


def multipop(li, n):
    if n > 0:
        li.pop()
        multipop(li, n - 1)
    return li


def optimize(c):
    iq = c[:]
    oq = []
    while len(iq):
        oq.append(iq.pop(0))
        if oq[-1] in ops and len(oq) >= 3:
            if isinstance(oq[-2], (int, long)) and isinstance(oq[-3], (int, long)):
                ntok = ops[oq[-1]](oq[-2], oq[-3])
                multipop(oq, 3).append(ntok)
        if oq[-1] == 'NOT' and len(oq) >= 2 and oq[-2] == 'NOT':
            multipop(oq, 2)
        if oq[-1] == 'ADD' and len(oq) >= 3 and oq[-2] == 0 and is_numberlike(oq[-3]):
            multipop(oq, 2)
        if oq[-1] in ['SUB', 'ADD'] and len(oq) >= 3 and oq[-3] == 0 and is_numberlike(oq[-2]):
            ntok = oq[-2]
            multipop(oq, 3).append(ntok)
    return oq


def compile_to_assembly(source, optimize_flag=1):
    if isinstance(source, (str, unicode)):
        source = parse(source)
    varhash = {}
    c1 = rewrite(source)
    c2 = compile_expr(c1, varhash, [0])
    c3 = add_wrappers(c2, varhash)
    c4 = optimize(c3) if optimize_flag else c3
    return c4


def get_vars(source):
    if isinstance(source, (str, unicode)):
        source = parse(source)
    varhash = {}
    c1 = rewrite(source)
    # fill varhash
    compile_expr(c1, varhash, [0])
    return varhash


def log256(n):
    return 0 if n == 0 else 1 + log256(n / 256)


def tobytearr(n, L):
    return [] if L == 0 else tobytearr(n / 256, L - 1) + [n % 256]


# Dereference labels
def dereference(c):
    iq = [x for x in c]
    mq = []
    pos = 0
    labelmap = {}
    while len(iq):
        front = iq.pop(0)
        if isinstance(front, str) and front[:6] == 'LABEL_':
            labelmap[front[6:]] = pos
        else:
            mq.append(front)
            if isinstance(front, str) and front[:4] == 'REF_':
                pos += 5
            elif isinstance(front, (int, long)):
                pos += 1 + max(1, log256(front))
            else:
                pos += 1
    oq = []
    for m in mq:
        if isinstance(m, str) and m[:4] == 'REF_':
            oq.append('PUSH4')
            oq.extend(tobytearr(labelmap[m[4:]], 4))
        elif isinstance(m, (int, long)):
            L = max(1, log256(m))
            oq.append('PUSH' + str(L))
            oq.extend(tobytearr(m, L))
        else:
            oq.append(m)
    return oq


def serialize(source):
    def numberize(arg):
        if isinstance(arg, (int, long)):
            return arg
        elif arg in reverse_opcodes:
            return reverse_opcodes[arg]
        elif arg[:4] == 'PUSH':
            return 95 + int(arg[4:])
        elif re.match('^[0-9]*$', arg):
            return int(arg)
        else:
            raise Exception("Cannot serialize: " + str(arg))
    return ''.join(map(chr, map(numberize, source)))


def deserialize(source):
    o = []
    i, j = 0, -1
    while i < len(source):
        p = ord(source[i])
        if j >= 0:
            o.append(p)
        elif p >= 96 and p <= 127:
            o.append('PUSH' + str(p - 95))
        else:
            o.append(opcodes[p][0])
        if p >= 96 and p <= 127:
            j = p - 95
        j -= 1
        i += 1
    return o


def assemble(asm):
    return serialize(dereference(asm))


def compile(source):
    return assemble(compile_to_assembly(parse(source)))


def encode_datalist(vals):
    def enc(n):
        if isinstance(n, (int, long)):
            return ''.join(map(chr, tobytearr(n, 32)))
        elif isinstance(n, (str, unicode)) and len(n) == 40:
            return '\x00' * 12 + n.decode('hex')
        elif isinstance(n, (str, unicode)):
            return '\x00' * (32 - len(n)) + n
        elif n is True:
            return 1
        elif n is False or n is None:
            return 0
    if isinstance(vals, (tuple, list)):
        return ''.join(map(enc, vals))
    elif vals == '':
        return ''
    else:
        # Assume you're getting in numbers or 0x...
        return ''.join(map(enc, map(numberize, vals.split(' '))))


def decode_datalist(arr):
    if isinstance(arr, list):
        arr = ''.join(map(chr, arr))
    o = []
    for i in range(0, len(arr), 32):
        o.append(frombytes(arr[i:i + 32]))
    return o
