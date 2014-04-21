import re

# Number of spaces at the beginning of a line
def spaces(ln):
    spaces = 0
    while spaces < len(ln) and ln[spaces] == ' ': spaces += 1
    return spaces

# Main parse function
def parse(document):
    return parse_lines(document.split('\n'))

def strip_line(ln):
    ln2 = ln.strip()
    if '//' in ln2:
        return ln2[:ln2.find('//')]
    else:
        return ln2

# Parse the statement-level structure, including if and while statements
def parse_lines(lns):
    o = []
    i = 0
    while i < len(lns):
        main = lns[i]
        # Skip empty lines
        if len(main.strip()) == 0:
            i += 1
            continue
        if spaces(main) > 0:
            raise Exception("Line "+str(i)+" indented too much!")
        main = strip_line(main)
        # Grab the child block of an if statement
        start_child_block = i+1
        indent = 99999999
        i += 1
        child_lns = []
        while i < len(lns):
            if len(strip_line(lns[i])) > 0:
                sp = spaces(lns[i])
                if sp == 0: break
                indent = min(sp,indent)
                child_lns.append(lns[i])
            i += 1
        child_block = map(lambda x:x[indent:],child_lns)
        # Calls parse_line to parse the individual line
        out = parse_line(main)
        # Include the child block into the parsed expression
        if out[0] in ['if', 'else', 'while', 'else if']:
            if len(child_block) == 0:
                raise Exception("If/else/while statement must have sub-clause! (%d)" % i)
            else:
                out.append(parse_lines(child_block))
        else:
            if len(child_block) > 0:
                raise Exception("Not an if/else/while statement, can't have sub-clause! (%d)" % i)
        # This is somewhat complicated. Essentially, it converts something like
        # "if c1 then s1 elif c2 then s2 elif c3 then s3 else s4" (with appropriate
        # indenting) to [ if c1 s1 [ if c2 s2 [ if c3 s3 s4 ] ] ]
        if out[0] == 'else if':
            if len(o) == 0: raise Exception("Cannot start with else if! (%d)" % i)
            u = o[-1]
            while len(u) == 4: u = u[-1]
            u.append(['if'] + out[1:])
        elif out[0] == 'else':
            if len(o) == 0: raise Exception("Cannot start with else! (%d)" % i)
            u = o[-1]
            while len(u) == 4: u = u[-1]
            u.append(out[1])
        else:
            # Normal case: just add the parsed line to the output
            o.append(out)
    return o[0] if len(o) == 1 else ['seq'] + o

# Tokens contain one or more chars of the same type, with a few exceptions
def chartype(c):
    if c in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.':
        return 'alphanum'
    elif c in '\t ': return 'space'
    elif c in '()[]': return 'brack'
    elif c == '"': return 'dquote'
    elif c == "'": return 'squote'
    else: return 'symb'

# Converts something like "b[4] = x+2 > y*-3" to
# [ 'b', '[', '4', ']', '=', 'x', '+', '2', '>', 'y', '*', '-', '3' ]
def tokenize(ln):
    tp = 'space'
    i = 0
    o = []
    global cur
    cur = ''
    # Finish a token and start a new one
    def nxt():
        global cur
        if len(cur) >= 2 and cur[-1] == '-':
            o.extend([cur[:-1],'-'])
        elif len(cur.strip()) >= 1:
            o.append(cur)
        cur = ''
    # Main loop
    while i < len(ln):
        c = chartype(ln[i])
        # Inside a string
        if tp == 'squote' or tp == "dquote":
            if c == tp:
                cur += ln[i]
                nxt()
                i += 1
                tp = 'space'
            elif ln[i:i+2] == '\\x':
                cur += ln[i+2:i+4].decode('hex')
                i += 4
            elif ln[i:i+2] == '\\n':
                cur += '\x0a'
                i += 2
            elif ln[i] == '\\':
                cur += ln[i+1]
                i += 2
            else:
                cur += ln[i]
                i += 1
        # Not inside a string
        else:
            if c == 'brack' or tp == 'brack': nxt()
            elif c == 'space': nxt()
            elif c != 'space' and tp == 'space': nxt()
            elif c == 'symb' and tp != 'symb': nxt()
            elif c == 'alphanum' and tp == 'symb': nxt()
            elif c == 'squote' or c == "dquote": nxt()
            cur += ln[i]
            tp = c
            i += 1
    nxt()
    if o[-1] in [':',':\n','\n']: o.pop()
    if tp in ['squote','dquote']: raise Exception("Unclosed string: "+ln)
    return o

# This is the part where we turn a token list into an abstract syntax tree
precedence = {
    '^': 1,
    '*': 2,
    '/': 3,
    '%': 4,
    '#/': 2,
    '#%': 2,
    '+': 3,
    '-': 3,
    '<': 4,
    '<=': 4,
    '>': 4,
    '>=': 4,
    '==': 5,
    'and': 6,
    '&&': 6,
    'or': 7,
    '||': 7,
    '!': 0
}

def toktype(token):
    if token is None: return None
    elif token in ['(','[']: return 'left_paren'
    elif token in [')',']']: return 'right_paren'
    elif token == ',': return 'comma'
    elif token == ':': return 'colon'
    elif token in ['!']: return 'unary_operation' 
    elif not isinstance(token,str): return 'compound'
    elif token in precedence: return 'binary_operation'
    elif re.match('^[0-9a-zA-Z\-\.]*$',token): return 'alphanum'
    elif token[0] in ['"',"'"] and token[0] == token[-1]: return 'alphanum'
    else: raise Exception("Invalid token: "+token)

# https://en.wikipedia.org/wiki/Shunting-yard_algorithm
#
# The algorithm works by maintaining three stacks: iq, stack, oq. Initially,
# the tokens are placed in order on the iq. Then, one by one, the tokens are
# processed. Values are moved immediately to the output queue. Operators are
# pushed onto the stack, but if an operator comes along with lower precendence
# then all operators on the stack with higher precedence are applied first.
# For example:
# iq = 2 + 3 * 5 + 7, stack = \, oq = \
# iq = + 3 * 5 + 7, stack = \, oq = 2
# iq = 3 * 5 + 7, stack = +, oq = 2
# iq = * 5 + 7, stack = +, oq = 2 3
# iq = 5 + 7, stack = + *, oq = 2 3 (since * > + in precedence)
# iq = + 7, stack = + *, oq = 2 3 5
# iq = 7, stack = + +, oq = 2 [* 3 5] (since + > * in precedence)
# iq = \, stack = + +, oq = 2 [* 3 5] 7
# iq = \, stack = +, oq = 2 [+ [* 3 5] 7]
# iq = \, stack = \, oq = [+ 2 [+ [* 3 5] 7] ]
#
# Functions, where function arguments begin with a left bracket preceded by
# the function name, are separated by commas, and end with a right bracket,
# are also included in this algorithm, though in a different way
def shunting_yard(tokens):
    iq = [x for x in tokens]
    oq = []
    stack = []
    prev,tok = None,None
    # The normal Shunting-Yard algorithm simply converts expressions into
    # reverse polish notation. Here, we try to be slightly more ambitious
    # and build up the AST directly on the output queue
    # eg. say oq = [ 2, 5, 3 ] and we add "+" then "*"
    # we get first [ 2, [ +, 5, 3 ] ] then [ [ *, 2, [ +, 5, 3 ] ] ]
    def popstack(stack,oq):
        tok = stack.pop()
        typ = toktype(tok)
        if typ == 'binary_operation':
            a,b = oq.pop(), oq.pop()
            oq.append([ tok, b, a])
        elif typ == 'unary_operation':
            a = oq.pop()
            oq.append([ tok, a ])
        elif typ == 'right_paren':
            args = []
            while toktype(oq[-1]) != 'left_paren':
                args.insert(0,oq.pop())
            oq.pop()
            if tok == ']' and args[0] != 'id':
                oq.append(['access'] + args)
            elif tok == ']':
                oq.append(['array_lit'] + args[1:])
            elif tok == ')' and len(args) and args[0] != 'id':
                oq.append(args)
            else:
                oq.append(args[1])
    # The main loop
    while len(iq) > 0:
        prev = tok
        tok = iq.pop(0)
        typ = toktype(tok)
        if typ == 'alphanum':
            oq.append(tok)
        elif typ == 'left_paren':
            # Handle cases like 3 * (2 + 5) by using 'id' as a default function
            # name
            if toktype(prev) != 'alphanum' and toktype(prev) != 'right_paren':
                oq.append('id')
            # Say the statement is "... f(45...". At the start, we would have f
            # as the last item on the oq. So we move it onto the stack, put the
            # leftparen on the oq, and move f back to the stack, so we have ( f
            # as the last two items on the oq. We also put the leftparen on the
            # stack so we have a separator on both the stack and the oq
            stack.append(oq.pop())
            oq.append(tok)
            oq.append(stack.pop())
            stack.append(tok)
        elif typ == 'right_paren':
            # eg. f(27, 3 * 5 + 4). First, we finish evaluating all the
            # arithmetic inside the last argument. Then, we run popstack
            # to coalesce all of the function arguments sitting on the
            # oq into a single list
            while len(stack) and toktype(stack[-1]) != 'left_paren':
                popstack(stack,oq)
            if len(stack):
                stack.pop()
            stack.append(tok)
            popstack(stack,oq)
        elif typ == 'unary_operation' or typ == 'binary_operation':
            # -5 -> 0 - 5
            if tok == '-' and toktype(prev) not in ['alphanum', 'right_paren']:
                oq.append('0')
            # Handle BEDMAS operator precedence
            prec = precedence[tok]
            while len(stack) and toktype(stack[-1]) == 'binary_operation' and precedence[stack[-1]] < prec:
                popstack(stack,oq)
            stack.append(tok)
        elif typ == 'comma':
            # Finish evaluating all arithmetic before the comma
            while len(stack) and toktype(stack[-1]) != 'left_paren':
                popstack(stack,oq)
        elif typ == 'colon':
            # Colon is like a comma except it stays in the argument list
            while len(stack) and toktype(stack[-1]) != 'right_paren':
                popstack(stack,oq)
            oq.append(tok)
    while len(stack):
        popstack(stack,oq)
    if len(oq) == 1:
        return oq[0]
    else:
        raise Exception("Wrong number of items left on stack: "+str(oq))

def parse_line(ln):
    tokens = tokenize(ln.strip())
    if tokens[0] == 'if' or tokens[0] == 'while':
        return [ tokens[0], shunting_yard(tokens[1:]) ]
    elif len(tokens) >= 2 and tokens[0] == 'else' and tokens[1] == 'if':
        return [ 'else if', shunting_yard(tokens[2:]) ]
    elif len(tokens) >= 1 and tokens[0] == 'elif':
        return [ 'else if', shunting_yard(tokens[1:]) ]
    elif len(tokens) == 1 and tokens[0] == 'else':
        return [ 'else' ]
    elif '=' in tokens:
        eqplace = tokens.index('=')
        return [ 'set', shunting_yard(tokens[:eqplace]), shunting_yard(tokens[eqplace+1:]) ]
    else:
        return shunting_yard(tokens)
