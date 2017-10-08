from __future__ import print_function, absolute_import, division, print_function, unicode_literals

import copy
import os
from typing import List

import sympy
import sympy.physics.mechanics as mech

from pymola import ast
from pymola.tree import TreeListener, TreeWalker, flatten

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
# noinspection PyUnresolvedReferences
BUILTINS = dir(__builtins__) + ['psi']


class DAE(object):
    def __init__(self, states, inputs, outputs, constants,
                 parameters, variables, equations, statements):
        self.x0 = {}  # type: Dict[sympy.Symbol, sympy.Expr]
        self.states = states  # type: List[sympy.Symbol]
        self.inputs = inputs  # type: List[sympy.Symbol]
        self.outputs = outputs  # type: List[sympy.Symbol]
        self.constants = constants  # type: List[sympy.Symbol]
        self.parameters = parameters  # type: List[sympy.Symbol]
        self.variables = variables  # type: List[sympy.Symbol]
        self.equations = equations  # type: List[sympy.Expr]
        self.statements = statements  # type: List[sympy.Expr]

    def __repr__(self):
        return repr(self.__dict__)


class SympyGenerator(TreeListener):
    def __init__(self):
        super(SympyGenerator, self).__init__()
        self.model = {}

    def exitFile(self, tree: ast.File):
        d = {}
        for key in tree.classes.keys():
            d[key] = self.model[tree.classes[key]]
        self.model[tree] = d

    def exitClass(self, tree: ast.Class):
        states = []
        inputs = []
        outputs = []
        constants = []
        parameters = []
        variables = []

        symbols = sorted(tree.symbols.values(), key=lambda x: x.order)  # type: List[ast.Symbol]

        for s in symbols:
            s_model = self.model[s]
            if len(s.prefixes) == 0:
                variables += [s_model]
            else:
                for prefix in s.prefixes:
                    if prefix == 'state':
                        states += [s_model]
                    elif prefix == 'constant':
                        constants += [s_model]
                    elif prefix == 'parameter':
                        parameters += [s_model]
                    elif prefix == 'input':
                        inputs += [s_model]
                    elif prefix == 'output':
                        outputs += [s_model]

        # each output that is not a state is a variable
        for s in outputs:
            if s not in states:
                variables += [s]

        equations = [self.model[eq] for eq in tree.equations]
        statements = [self.model[stmt] for stmt in tree.statements]

        self.model[tree] = DAE(
            states=states,
            inputs=inputs,
            outputs=outputs,
            constants=constants,
            parameters=parameters,
            variables=variables,
            equations=equations,
            statements=statements)

    def exitExpression(self, tree: ast.Expression):
        op = str(tree.operator)
        n_operands = len(tree.operands)
        t = sympy.Symbol('t')
        model = None
        if op == 'der':
            model = self.model[tree.operands[0]].diff(t)
        elif n_operands == 2:
            left = self.model[tree.operands[0]]
            right = self.model[tree.operands[1]]
            model = eval("left {:s} right".format(op))
        elif n_operands == 1:
            right = self.model[tree.operands[0]]
            try:
                model = eval("{:s}(right)".format(op))
            except NameError:
                model = eval("sympy.{:s}(right)".format(op))
        else:
            raise SyntaxError('unknown operator {:s}'.format(op))
        self.model[tree] = model

    def exitPrimary(self, tree: ast.Primary):
        self.model[tree] = sympy.sympify(tree.value)

    def exitComponentRef(self, tree: ast.ComponentRef):

        # prevent name clash with builtins
        name = tree.name.replace('.', '__')
        while name in BUILTINS:
            name = name + '_'
        self.model[tree] = mech.dynamicsymbols(name)

    def exitSymbol(self, tree: ast.Symbol):
        # prevent name clash with builtins
        name = tree.name.replace('.', '__')
        while name in BUILTINS:
            name = name + '_'
        symbol = mech.dynamicsymbols(name)
        if tree.class_modification in self.model:
            # TODO need to store somewhere
            pass
        self.model[tree] = symbol

    def exitEquation(self, tree: ast.Equation):
        self.model[tree] = self.model[tree.left] - self.model[tree.right]

    def exitFunction(self, tree: ast.Function):
        f = sympy.Function(tree.name)
        self.model[tree] = f(*tuple([self.model[a] for a in tree.args]))

    def exitIfStatement(self, tree: ast.IfStatement):
        blocks = [[self.model[e] for e in b] for b in tree.blocks]
        conditions = [self.model[c] for c in tree.conditions]
        pairs = list(zip(blocks, conditions))
        self.model[tree] = sympy.Piecewise(*pairs)

    def exitIfEquation(self, tree: ast.IfEquation):
        blocks = [[self.model[e] for e in b] for b in tree.blocks]
        conditions = [self.model[c] for c in tree.conditions]
        pairs = list(zip(blocks, conditions))
        self.model[tree] = sympy.Piecewise(*pairs)

    def exitAssignmentStatement(self, tree: ast.AssignmentStatement):
        # more than one left symbol not yet supported
        assert len(tree.left) == 1
        self.model[tree] = self.model[tree.left[0]] - self.model[tree.right]

    def enterEvery(self, tree: ast.Node):
        # print('enter', tree.__class__.__name__)
        pass

    def exitEvery(self, tree: ast.Node):
        # print('exit', tree.__class__.__name__)
        pass

    def exitWhenEquation(self, tree: ast.WhenEquation):
        blocks = [[self.model[e] for e in b] for b in tree.blocks]
        conditions = [self.model[c] for c in tree.conditions]
        pairs = list(zip(blocks, conditions))
        self.model[tree] = sympy.Piecewise(*pairs)

    def exitWhenStatement(self, tree: ast.WhenStatement):
        def exitIfEquation(self, tree: ast.IfEquation):
            blocks = [[self.model[e] for e in b] for b in tree.blocks]
            conditions = [self.model[c] for c in tree.conditions]
            pairs = list(zip(blocks, conditions))
            self.model[tree] = sympy.Piecewise(*pairs)

    def exitForEquation(self, tree: ast.ForEquation):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitForIndex(self, tree: ast.ForIndex):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitArray(self, tree: ast.Array):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitClassModification(self, tree: ast.ClassModification):
        self.model[tree] = [self.model[a] for a in tree.arguments]

    def exitComponentClause(self, tree: ast.ComponentClause):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitConnectClause(self, tree: ast.ConnectClause):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitSlice(self, tree: ast.Slice):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitElementModification(self, tree: ast.ElementModification):
        self.model[tree] = {tree.component: tree.modifications[0].value}

    def exitExtendsClause(self, tree: ast.ExtendsClause):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitForStatement(self, tree: ast.ForStatement):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitIfExpression(self, tree: ast.IfExpression):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitImportAsClause(self, tree: ast.ImportAsClause):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))

    def exitImportFromClause(self, tree: ast.ImportFromClause):
        # TODO
        raise NotImplementedError(
            "{:s} not yet handled in sympy generator".format(tree.__class__.__name__))


def generate(ast_tree: ast.Collection, model_name: str):
    """
    :param ast_tree: AST to generate from
    :param model_name: class to generate
    :return: sympy source code for model
    """
    component_ref = ast.ComponentRef.from_string(model_name)
    ast_tree_new = copy.deepcopy(ast_tree)
    ast_walker = TreeWalker()
    flat_tree = flatten(ast_tree_new, component_ref)
    sympy_gen = SympyGenerator()
    ast_walker.walk(sympy_gen, flat_tree)
    return sympy_gen.model[flat_tree][model_name]
