"""Bounded typed derivations and compilation into executable equation DAGs."""
from collections import deque
from dataclasses import dataclass


AXES = ('D_x', 'D_y', 'D_z')
# Result type, child types, and spatial derivative order added on a path.
RULES = {
    'u': ('s', (), 0), 'u_t': ('s', (), 0), 'q': ('s', (), 0), 'f': ('s', (), 0),
    **{axis: ('s', ('s',), 1) for axis in AXES},
    'lap': ('s', ('s',), 2), 'add': ('s', ('s', 's'), 0),
    'mul': ('s', ('s', 's'), 0), 'square': ('s', ('s',), 0),
    'cube': ('s', ('s',), 0), 'sin': ('s', ('s',), 0), 'tanh': ('s', ('s',), 0),
    'vec': ('v', ('s', 's', 's'), 0), 'grad': ('v', ('s',), 1),
    'scale': ('v', ('s', 'v'), 0), 'dot': ('s', ('v', 'v'), 0),
    'div': ('s', ('v',), 1),
}
COUNTS = ('1 branch', '2 branches', '3 branches', '4 branches')
TOKENS = ('[MASK]', *COUNTS, *RULES)
INDEX = {token: i for i, token in enumerate(TOKENS)}
MAX_OPS = 8
MAX_SLOTS = 1 + 4 * (1 + 3 * MAX_OPS)


@dataclass(frozen=True)
class Slot:
    branch: int
    kind: str
    order: int
    parent: int


class Derivation:
    """Breadth-first expansion. Pending vector slots reserve one operator each."""

    def __init__(self, leaves=('u', 'u_t')):
        self.leaves = frozenset(leaves)
        if 'u' not in self.leaves or not self.leaves <= {'u', 'u_t', 'q', 'f'}:
            raise ValueError('The grammar requires u and accepts u_t, q, and f.')
        self.tokens = []
        self.children = []
        self.pending = deque([Slot(-1, 'root', 2, -1)])
        self.used = [0] * 4
        self.reserved = [0] * 4

    def legal(self):
        slot = self.pending[0]
        if slot.kind == 'root':
            return COUNTS
        branch = slot.branch
        committed = self.used[branch] + self.reserved[branch] - (slot.kind == 'v')
        return tuple(rule for rule, (kind, args, cost) in RULES.items()
                     if kind == slot.kind and cost <= slot.order
                     and (args or rule in self.leaves)
                     and committed + bool(args) + args.count('v') <= MAX_OPS)

    def add(self, rule):
        if rule not in self.legal():
            raise ValueError(f'Illegal production {rule!r} at {self.pending[0]}')
        slot = self.pending.popleft()
        i = len(self.tokens)
        self.tokens.append(rule)
        self.children.append([])
        if slot.kind == 'root':
            self.pending.extend(Slot(j, 's', 2, i) for j in range(COUNTS.index(rule) + 1))
        else:
            self.children[slot.parent].append(i)
            _, args, cost = RULES[rule]
            branch = slot.branch
            self.used[branch] += bool(args)
            self.reserved[branch] += args.count('v') - (slot.kind == 'v')
            self.pending.extend(Slot(branch, kind, slot.order - cost, i) for kind in args)

    def equation(self):
        if self.pending:
            raise ValueError('Complete the derivation before compiling it.')
        nodes = [None] * len(self.tokens)
        for i in range(len(nodes) - 1, 0, -1):
            nodes[i] = (self.tokens[i], *(nodes[j] for j in self.children[i]))
        return canonical_graph(nodes[i] for i in self.children[0])


def expression(op, *args):
    return (op, *args)


U = expression('u')
LAPLACIAN = expression('lap', U)
MIXED = expression('D_x', expression('D_y', U))
PURE = tuple(expression(axis, expression(axis, U)) for axis in AXES)


def _flatten(op, expr):
    if expr[0] == op:
        return _flatten(op, expr[1]) + _flatten(op, expr[2])
    return [expr]


def _canonical_step(expr):
    op, *raw = expr
    args = tuple(_canonical_step(arg) for arg in raw)
    if op in ('add', 'mul'):
        terms = sorted(term for arg in args for term in _flatten(op, arg))
        if op == 'mul':
            if terms[0] == terms[1] and len(terms) == 2:
                return ('square', terms[0])
            if len(terms) == 3 and terms[0] == terms[2]:
                return ('cube', terms[0])
            for i in (0, 1):
                a, b = args[i], args[1 - i]
                if a == ('square', b):
                    return ('cube', b)
        elif len(terms) == 3:
            inner = [term[1][1] for term, axis in zip(terms, AXES)
                     if term[0] == axis and term[1][0] == axis]
            if len(inner) == 3 and inner[0] == inner[1] == inner[2]:
                return ('lap', inner[0])
        result = terms[-1]
        for term in reversed(terms[:-1]):
            result = (op, term, result)
        return result
    if op in AXES and args[0][0] in AXES and op > args[0][0]:
        inner_op, child = args[0]
        return (inner_op, (op, child))
    if op == 'div' and args[0][0] == 'grad':
        return ('lap', args[0][1])
    if op == 'dot':
        args = tuple(sorted(args))
    return (op, *args)


def canonical(expr):
    """Normalize common grammar identities without fitting or inspecting data."""
    simplified = _canonical_step(expr)
    while simplified != expr:
        expr, simplified = simplified, _canonical_step(simplified)
    return simplified


def canonical_graph(branches):
    return tuple(sorted(set(canonical(branch) for branch in branches)))


def encode(graph, leaves):
    state = Derivation(leaves)
    state.add(COUNTS[len(graph) - 1])
    queue = deque(graph)
    while queue:
        op, *args = queue.popleft()
        state.add(op)
        queue.extend(args)
    return state.tokens


def compile_dag(graph):
    """Nodes are topologically ordered and keep ordered, possibly repeated inputs."""
    nodes = {}

    def visit(expr):
        op, *args = expr
        key = op, tuple(visit(arg) for arg in args)
        if key not in nodes:
            nodes[key] = len(nodes)
        return nodes[key]

    roots = tuple(visit(branch) for branch in graph)
    return tuple(nodes), roots


def equation_dag(graph):
    """Include the independent coefficients, branch scaling and RHS assembly."""
    expressions, roots = compile_dag(graph)
    nodes = list(expressions)
    scaled = []
    for branch, root in enumerate(roots, 1):
        coefficient = len(nodes)
        nodes.append((f'theta_{branch}', ()))
        scaled.append(len(nodes))
        nodes.append(('mul', (coefficient, root)))
    output = scaled[0]
    for node in scaled[1:]:
        nodes.append(('add', (output, node)))
        output = len(nodes) - 1
    return {'nodes': nodes, 'rhs': output, 'target': 'u_tt'}


def complexity(graph):
    nodes, _ = compile_dag(graph)
    # One multiply per fitted coefficient and m-1 summation operators.
    return sum(bool(children) for _, children in nodes) + 3 * len(graph) - 1


def format_expr(expr):
    op, *args = expr
    children = [format_expr(arg) for arg in args]
    if op in ('add', 'mul'):
        symbol = '+' if op == 'add' else '·'
        return '(' + symbol.join(children) + ')'
    labels = {'lap': 'Δ', 'grad': '∇', 'div': 'div', 'D_x': 'Dₓ', 'D_y': 'Dᵧ', 'D_z': 'D𝓏'}
    return labels.get(op, op) + (f'({", ".join(children)})' if args else '')
