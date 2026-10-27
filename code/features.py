"""Reference-free, permutation-equivariant candidate evidence."""
import re
import ast
import operator
from fractions import Fraction
from itertools import permutations
from functools import lru_cache
import numpy as np
from common import parse_answer

WORDS = ('total', 'remaining', 'left', 'each', 'per', 'more', 'less', 'difference',
         'original', 'discount', 'percent', 'half', 'twice', 'average', 'rate',
         'however', 'but', 'assume', 'approximately', 'round', 'check', 'wait',
         'not', 'instead', 'therefore', 'correct', 'mistake', 'answer', 'step')
NUM = re.compile(r'(?<![a-zA-Z])[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')


def text_features(bank):
    result, equal = [], []
    for row in bank:
        texts = [c['text'] for c in row['candidates']]
        ans = [parse_answer(t) for t in texts]
        eq = np.array([[a is not None and a == b for b in ans] for a in ans], np.float32)
        equal.append(eq)
        q = row['question'].lower()
        qnums = set(NUM.findall(q))
        feats = []
        for t, a, e in zip(texts, ans, eq):
            low = t.lower()
            nums = NUM.findall(t)
            try:
                from fractions import Fraction
                value = float(Fraction(a)) if a is not None else 0.
            except (OverflowError, ValueError):
                value = 0.
            f = [a is not None, e.sum()/4, len(set(x for x in ans if x is not None))/4,
                 np.log1p(len(t)), np.log1p(len(t.split())), np.log1p(len(nums)),
                 np.log1p(t.count('\n')), t.count('=')/10, t.count('\\frac')/10,
                 t.count('\\boxed')/5, t.count('####')/5, t.count('**')/20,
                 t.count('?')/5, np.sign(value), np.log1p(abs(value)),
                 value == int(value) if np.isfinite(value) else 0,
                 a in qnums, len(set(nums)&qnums)/max(1,len(qnums)),
                 np.log1p(len(q)), len(qnums)/10]
            f.extend(low.count(w)/max(1,len(t.split()))*10 for w in WORDS)
            f.extend(float(w in q) for w in WORDS)
            f.extend(float(w in q)*low.count(w)/max(1,len(t.split()))*10 for w in WORDS)
            feats.append(f)
        f = np.asarray(feats, np.float32)
        result.append(np.concatenate([f, f-f.mean(0,keepdims=True)],axis=1))
    return np.asarray(result,np.float32), np.asarray(equal,np.float32)


def build_features(bank, arrays, mode='full'):
    if mode == 'answer':
        stripped=[{'question':r['question'],'candidates':[{'text':'#### '+(parse_answer(c['text']) or '')} for c in r['candidates']]} for r in bank]
        return text_features(stripped)
    text, equal = text_features(bank)
    e = arrays['embeddings'].astype(np.float32)
    norm=np.linalg.norm(e,axis=-1,keepdims=True)
    if mode not in ('raw','raw_interaction'):
        e = e / np.maximum(norm,1e-8) * 32
    if mode == 'style':
        x = text
    elif mode == 'embedding':
        x = e
    elif mode == 'relative':
        x = np.concatenate([e,e-e.mean(1,keepdims=True)],axis=-1)
    else:
        x = np.concatenate([e,e-e.mean(1,keepdims=True),text],axis=-1)
        if mode in ('interaction','raw_interaction'):
            context=e.mean(1,keepdims=True)
            x=np.concatenate([x,e*context],axis=-1)
        if mode == 'enhanced':
            x=np.concatenate([x,extra_features(bank,e,equal)],axis=-1)
        if mode=='norm':
            x=np.concatenate([x,norm,norm-norm.mean(1,keepdims=True)],axis=-1)
        if mode=='relations':
            x=np.concatenate([x,norm,norm-norm.mean(1,keepdims=True),relation_features(bank)],axis=-1)
    return np.asarray(x,np.float32),equal


REL_WORDS=('total','left','remaining','profit','difference','average','percent','discount','more','less','each','per',
           'distance','time','minutes','hours','days','twice','half','together','earn','pay','cost','spent','sell','buy')
NUMBER_WORDS={'one':'1','two':'2','three':'3','four':'4','five':'5','six':'6','seven':'7','eight':'8','nine':'9',
              'ten':'10','eleven':'11','twelve':'12','dozen':'12','twenty':'20','thirty':'30','sixty':'60'}

@lru_cache(maxsize=12)
def tuples_of_numbers(n):
    return (np.array(list(permutations(range(n),2)),np.int64).reshape(-1,2),
            np.array(list(permutations(range(n),3)),np.int64).reshape(-1,3))


def relation_features(bank):
    """Candidate answer compatibility with generic arithmetic relations in the question.

    All recipes are fixed, reference-free, and apply to arbitrary new questions.
    They supply evidence to a learned verifier; they never replace a candidate answer.
    """
    rows=[]
    for row in bank:
        q=row['question'].lower().replace(',','')
        qn=re.sub(r'\b('+'|'.join(NUMBER_WORDS)+r')\b',lambda m:NUMBER_WORDS[m[0]],q)
        nums=[]
        for match in re.finditer(r'\d+(?:\.\d+)?(?:/\d+)?',qn):
            try:nums.append(float(Fraction(match[0])))
            except (ValueError,ZeroDivisionError):pass
        nums=np.array(nums[:8],np.float64)
        if len(nums)==0:nums=np.array([0.])
        pair,triple=tuples_of_numbers(len(nums));a,b=nums[pair].T if len(pair) else (np.array([]),np.array([]))
        aa,bb,cc=nums[triple].T if len(triple) else (np.array([]),np.array([]),np.array([]))
        with np.errstate(divide='ignore',invalid='ignore',over='ignore'):
            recipes=[nums,a+b,a-b,a*b,a/b,a*b/100,a*100/b,a*(1-b/100),a/(1-b/100),a*(1+b/100),a/(1+b/100),
                     aa+bb+cc,aa+bb-cc,(aa+bb)*cc,(aa-bb)*cc,aa*bb+cc,aa*bb-cc,aa*bb/cc,
                     aa/(bb+cc),aa/(bb-cc),(aa+bb)/cc,(aa-bb)/cc,aa+bb/cc,aa-bb/cc,aa*bb*cc,
                     aa*(bb-cc)/100,aa*100/(bb-cc),np.array([nums.sum(),nums.mean(),nums.prod()])]
        recipes=[r[np.isfinite(r)&(abs(r)<1e12)] for r in recipes]
        flags=np.array([float(w in q) for w in REL_WORDS]+[float(w in q[-100:]) for w in REL_WORDS],np.float32)
        group=[]
        for c in row['candidates']:
            ans=parse_answer(c['text'])
            try:value=float(Fraction(ans)) if ans is not None else 1e15
            except (OverflowError,ValueError):value=1e15
            errors=np.array([min(1e6,float(np.min(abs(r-value)/(1+abs(value))))) if len(r) else 1e6 for r in recipes])
            near=np.exp(-10*errors).astype(np.float32)
            # Fixed arithmetic evidence interacts with generic question conditions.
            feature=np.r_[near, -np.log1p(errors).clip(0,10)/10,(errors<1e-6).astype(np.float32),
                          (near[:,None]*flags[None,:]).ravel()]
            group.append(feature)
        f=np.array(group,np.float32);rows.append(np.concatenate([f,f-f.mean(0)],axis=-1))
    return np.asarray(rows,np.float32)


def arithmetic_value(expression):
    ops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
    def visit(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):return node.value
        if isinstance(node,ast.BinOp) and type(node.op) in ops:return ops[type(node.op)](visit(node.left),visit(node.right))
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.USub,ast.UAdd)):
            return (-1 if isinstance(node.op,ast.USub) else 1)*visit(node.operand)
        raise ValueError('Not a numeric arithmetic expression')
    if len(expression)>150:raise ValueError('Expression too long')
    return float(visit(ast.parse(expression.strip(),mode='eval').body))


def extra_features(bank,emb,equal):
    rows=[]
    for row,e,eq in zip(bank,emb,equal):
        qwords=set(re.findall(r'[a-z]+',row['question'].lower()))
        group=[]
        for i,c in enumerate(row['candidates']):
            t=c['text'];clean=t.replace(',','')
            for _ in range(4):
                clean=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',clean)
            clean=re.sub(r'\\boxed\{([^{}]+)\}',r'\1',clean)
            clean=clean.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('×','*').replace('÷','/').replace('−','-')
            clean=re.sub(r'\\(?:text|mathrm)\{[^{}]*\}',' ',clean).replace('**','')
            good=bad=0;errors=[]
            for line in clean.splitlines():
                for left,right in re.findall(r'([\d.()+*/ \t-]+)=\s*([-+]?\d+(?:\.\d+)?(?:/\d+)?)',line):
                    if not re.search(r'\d\s*[+*/-]',left):continue
                    try:
                        lv=arithmetic_value(left);rv=arithmetic_value(right)
                        err=abs(lv-rv)/max(1,abs(lv),abs(rv))
                        good+=err<1e-5;bad+=err>=1e-5;errors.append(err)
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
            words=set(re.findall(r'[a-z]+',t.lower()))
            numbers=NUM.findall(t)
            answer=parse_answer(t)
            sims=e@e[i]/1024
            others=eq[i].copy();others[i]=0
            final_words=set(re.findall(r'[a-z]+',t[-200:].lower()))
            group.append([good/10,bad/10,good/max(1,good+bad),np.mean(errors) if errors else 0,
                          len(qwords&words)/max(1,len(qwords)),len(qwords&final_words)/max(1,len(qwords)),
                          len(words)/max(1,len(t.split())),len(set(numbers))/max(1,len(numbers)),
                          numbers.count(answer)/max(1,len(numbers)) if answer else 0,
                          (others@sims)/max(1,others.sum()),sims.mean(),sims.min(),sims.std(),
                          float('####' in t),float('\\boxed' in t),
                          np.log1p(len(t.rsplit('####',1)[-1])) if '####' in t else 0])
        f=np.array(group,np.float32);rows.append(np.concatenate([f,f-f.mean(0)],axis=-1))
    return np.asarray(rows,np.float32)


def aggregate(scores, equal, weight=0):
    """Blend candidate evidence with mean evidence for the same numeric answer."""
    means = (equal @ scores[...,None])[...,0] / np.maximum(equal.sum(-1),1)
    return (1-weight)*scores + weight*means


def fusion_features(family_scores,equal):
    """Shared candidate features for an OOF-trained disagreement-aware fusion."""
    s=np.stack(family_scores,axis=-1).astype(np.float32)
    count=equal.sum(-1,keepdims=True)
    grouped=(equal@s)/np.maximum(count,1)
    unique=(1/np.maximum(count,1)*(count>0)).sum(1,keepdims=True)
    return np.concatenate([s,s-s.mean(1,keepdims=True),grouped,s-s.max(1,keepdims=True),
                           count/4,np.broadcast_to(unique/4,count.shape)],axis=-1)


def tree_predict(x,w):
    flat=x.reshape(-1,x.shape[-1]);out=np.zeros(len(flat),np.float32)
    for fi,border,leaf in zip(w['features'],w['borders'],w['leaves']):
        index=((flat[:,fi]>border)*np.array([1,2,4],np.int32)).sum(-1)
        out+=leaf[index]
    return (out*w['scale_tree']+w['bias_tree']).reshape(x.shape[:2])


def arithmetic_evidence(bank):
    """Check numeric equality chains, including fractional right-hand sides.

    Symbolic expressions are skipped. Rounded decimal steps receive tolerance.
    The result is evidence rather than a correctness guarantee.
    """
    result=[]
    for row in bank:
        features=[]
        for c in row['candidates']:
            t=c['text'].replace(',','').replace('**','').replace('$','')
            for _ in range(6):t=re.sub(r'\\(?:d?frac)\{([^{}]+)\}\{([^{}]+)\}',r'((\1)/(\2))',t)
            t=re.sub(r'\\boxed\{([^{}]+)\}',r'\1',t)
            t=re.sub(r'\\(?:text|mathrm)\{[^{}]*\}',' ',t)
            t=t.replace('\\times','*').replace('\\cdot','*').replace('\\div','/').replace('×','*').replace('÷','/').replace('−','-')
            t=t.replace('\\approx','=');errors=[]
            for line in t.splitlines():
                line=re.sub(r'^\s*[-*>#]\s+','',line)
                parts=line.split('=')
                for left,right in zip(parts,parts[1:]):
                    lm=re.search(r'([+\-]?[\d.(][\d.()+*/ \t-]*)\s*$',left)
                    rm=re.match(r'\s*([+\-]?[\d.(][\d.()+*/ \t-]*)',right)
                    if lm is None or rm is None:continue
                    le,rex=lm[1].strip(),rm[1].strip()
                    if not re.search(r'[+*/-]',le+rex):continue
                    try:
                        lv=arithmetic_value(le);rv=arithmetic_value(rex)
                        if not np.isfinite(lv+rv):continue
                        errors.append(abs(lv-rv)/max(1,abs(lv),abs(rv)))
                    except (ValueError,SyntaxError,ZeroDivisionError,OverflowError):pass
            err=np.asarray(errors)
            features.append([len(err),int((err>1e-3).sum()),int((err>.01).sum()),
                             float(err.max()) if len(err) else 0.,float(err.mean()) if len(err) else 0.])
        result.append(features)
    return np.array(result,np.float32)
