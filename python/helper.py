const = lambda expr: (expr.count(' ') == 2 and not ('∃' in expr or '∀' in expr))
quant = lambda expr:  (not ('⊔' in expr or '⊓' in expr)) and ('∃' in expr or '∀' in expr)


def atomic_check(exp):
    return ' ' not in exp

def neg_check(exp):
    return (' ' not in exp) and ('¬' in exp)

def three_check(exp):
    return quant(exp) or const(exp)

def inter_check(expr):
    return expr.count(' ') == 2 and ('⊓' in expr) and ('¬' not in expr) and top_bot(expr)

def union_check(expr):
    return expr.count(' ') == 2 and ('⊔' in expr) and ('¬' not in expr) and top_bot(expr)

def exist_check(expr):
    return quant(expr) and ('∃' in expr) and ('¬' not in expr) and top_bot(expr)

def forall_check(expr):
    return quant(expr) and ('∀' in expr) and ('¬' not in expr) and top_bot(expr)

def null(expr):
    return False

def top_bot(exp):
    return (('⊤' not in exp) and ('⊥' not in exp))