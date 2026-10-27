"""Conservative numeric checks with percentages, powers and local variables.

No reference answers or problem IDs are used. Expressions are interpreted with
a bounded AST visitor; unsupported syntax is skipped. A failed equality is only
evidence against a trace, not a proof that its final answer is incorrect.
"""
import ast
import operator
import re
import numpy as np


def value(expression,environment):
    operations={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.Name) and node.id in environment:return environment[node.id]
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return (-1 if isinstance(node.op,ast.USub) else 1)*visit(node.operand)
        if isinstance(node,ast.BinOp):
            a,b=visit(node.left),visit(node.right)
            if type(node.op) in operations:return operations[type(node.op)](a,b)
            if isinstance(node.op,ast.Pow) and abs(a)<=1e8 and abs(b)<=6:return a**b
        raise ValueError('Unsupported arithmetic')
    if len(expression)>150:raise ValueError('Long expression')
    result=float(visit(ast.parse(expression,mode='eval').body))
    if not np.isfinite(result) or abs(result)>1e15:raise ValueError('Unbounded value')
    return result


def clean(text):
    text=text.replace(',','').replace('**','').replace('$','')
    for _ in range(8):
        text=re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}',r'((\1)/(\2))',text)
    text=re.sub(r'\\boxed\s*\{([^{}]+)\}',r'\1',text)
    text=re.sub(r'\\(?:text|mathrm|textrm)\s*\{[^{}]*\}',' ',text)
    for a,b in [('\\times','*'),('\\cdot','*'),('\\div','/'),('×','*'),('÷','/'),('−','-'),('\\approx','='),('≈','=')]:
        text=text.replace(a,b)
    text=re.sub(r'\\(?:left|right)','',text)
    text=re.sub(r'\\[,!; ]',' ',text)
    text=re.sub(r'\^\{([+\-]?\d+)\}',r'**(\1)',text)
    text=re.sub(r'\^(?=[+\-]?\d)',r'**',text)
    text=text.replace('\\%','%')
    text=re.sub(r'%\s+of\s+(?=[\d.(])','% * ',text)
    text=text.replace('%','/100')
    text=re.sub(r'(?<=\d)\s*[xX]\s*(?=\d)','*',text)
    # Explicit adjacent numeric parentheses denote multiplication.
    text=re.sub(r'(?<=[\d)])\s*(?=\()', '*',text)
    return text


def edge_expression(part,side,symbolic):
    # Prose is a delimiter. Only single-letter variables are accepted.
    part=re.sub(r'\b[A-Za-z_]\w*\b',lambda m:m[0] if symbolic and len(m[0])==1 else '|',part)
    pattern=r'[A-Za-z0-9_.()+*/ \t-]+' if symbolic else r'[0-9.()+*/ \t-]+'
    match=re.search('('+pattern+r')\s*$',part) if side=='left' else re.match(r'\s*('+pattern+')',part)
    if match is None:return ''
    return match[1].strip().rstrip('.').strip()


def evidence(bank):
    output=[]
    for row in bank:
        group=[]
        for candidate in row['candidates']:
            numeric=[];symbolic=[];environment={}
            for line in clean(candidate['text']).splitlines():
                line=re.sub(r'^\s*[-*>#]\s+','',line)
                parts=line.split('=')
                for left,right in zip(parts,parts[1:]):
                    le,rex=edge_expression(left,'left',True),edge_expression(right,'right',True)
                    nl,nr=edge_expression(left,'left',False),edge_expression(right,'right',False)
                    try:
                        # Never treat the numeric tail of an algebraic term
                        # (e.g. x + 1) as a standalone equality.
                        if re.search(r'[A-Za-z]',le+rex):raise ValueError('Symbolic equation')
                        if not re.search(r'[+*/-]',nl+nr):raise ValueError('No calculation')
                        a,b=value(nl,{}),value(nr,{})
                        numeric.append(abs(a-b)/max(1,abs(a),abs(b)))
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,TypeError):pass
                    try:
                        b=value(rex,environment)
                        # A bare variable defines or updates local state; it is
                        # not an inconsistency merely because it is reassigned.
                        if re.fullmatch(r'[A-Za-z]',le):environment[le]=b;continue
                        if not re.search(r'[A-Za-z]',le+rex):continue
                        a=value(le,environment)
                        symbolic.append(abs(a-b)/max(1,abs(a),abs(b)))
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,TypeError):pass
            err=np.asarray(numeric);se=np.asarray(symbolic)
            group.append([len(err),int((err>1e-3).sum()),int((err>.01).sum()),int((se>.01).sum())])
        output.append(group)
    return np.asarray(output,np.float32)
