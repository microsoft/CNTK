from .context import get_context

class ComputationNode(object):
    def __init__(self, name, params=None, ctx=None):
        self.name = name
        self.params = params
        self.var_name = None
        # context is used to get the graph, readers, etc.
        self.context = ctx or get_context()

        if self._is_feature():
            self.context.graph.add_feature(self)

        if self._is_label():
            self.context.graph.add_label(self)

    def _is_feature(self):
        return hasattr(self, 'tag') and self.tag == 'feature'

    def _is_label(self):
        return hasattr(self, 'tag') and self.tag == 'label'

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(other, self)

    def __sub__(self, other):
        return Minus(self, other)

    def __rsub__(self, other):
        return Minus(other, self)

    def __mul__(self, other):
        return ElementTimes(self, other)

    def __rmul__(self, other):
        return ElementTimes(other, self)

    def __matmul__(self, other):
        # NOTE supported in Python 3.5
        return Times(self, other)

    def __rmatmul__(self, other):
        # NOTE supported in Python 3.5
        return Times(other, self)

    def __abs__(self):
        return Abs(self)

    def __getitem__(self, so):
        if so.stop == None:
            raise ValueError('The stop index has to be provided')

        if isinstance(so, int):
            return RowSlice(self, so, 1)

        elif isinstance(so, slice):
            if so.step not in {1, None}:
                raise ValueError("RowSlice does not support strides")

            start = so.start or 0

            return RowSlice(self, start, so.stop - start)

    # TODO more __operators__

    def _get_cntk_param_string(self, param_variable_names=None):
        return ", ".join(param_variable_names)

    def eval(self, input_map, **kw):
        graph.eval(self, input_map, kw)

    def __str__(self):
        return "%s / params=%s"%(self.name, self.params)

    def _param_to_brainscript(self, p_name, p_value):
        if isinstance(p_value, bool):
            p_value = str(p_value).lower()
        elif isinstance(p_value, str):
            p_value = "'%s'"%p_value
        elif type(p_value) in [list, tuple]:
            # FIXME here we assume that all dims are of TensorShape
            if p_name=='dims':
                p_value = ":".join(v for v in p_value)
            else:
                raise ValueError('Sequence initialization is only allowed for'+
                 ' parameter "dims" and not "%s"'%p_name)
        else:
            p_value = str(p_value)

        return p_value

    def _to_description_unroll(self, desc, unrolled_nodes, node_counter=0):
        param_variable_names = []
        if self.params:
            for p_name in self.params:
                p_value = self.__dict__[p_name]
                if hasattr(p_value, '_to_description') and p_name:
                    if p_value in unrolled_nodes:
                        # we have seen this node already, so just retrieve its
                        # name
                        child_var = unrolled_nodes[p_value]
                    else:
                        child_var, node_counter, child_desc = p_value._to_description_unroll(desc, unrolled_nodes, node_counter)
                        unrolled_nodes[p_value] = child_var
                    param_variable_names.append(child_var)
                else:
                    param_variable_names.append(self._param_to_brainscript(p_name, p_value))

        self.var_name = self.var_name or "v%i"%node_counter 
        node_counter += 1
        
        params = self._get_cntk_param_string(param_variable_names)

        line = "%s = %s(%s)"%(self.var_name, self.name, params)
        desc.append(line)

        return self.var_name, node_counter, desc

    def _to_description(self):
        unrolled_nodes = {}
        var_name, node_counter, desc = self._to_description_unroll(desc=[], unrolled_nodes=unrolled_nodes)

        return var_name, node_counter, desc 

    def to_description(self):
        var_name, node_counter, desc = self._to_description()

        return "\n".join(desc)

    def to_graph_description(self, dummy_required=False):
        var_name, node_counter, desc = self._to_description()

        # FIXME we currently assume that the last node is also the root node -
        feature_nodes = self.context.graph.feature_nodes.copy()

        if dummy_required:
            var_name += ',dummy_node'
            feature_nodes.add('dummy_node')

        desc.append("OutputNodes=(%s)"%var_name)
        if feature_nodes:
            desc.append("FeatureNodes=(%s)"%','.join(node.var_name for node in feature_nodes))

        return "\n".join(desc)

class Label(ComputationNode):
    def __init__(self, dims, ctx=None):
        super(Label, self).__init__('Input', params=('dims', 'tag'), ctx=ctx)
        self.dims = dims
        self.tag = 'label'

class Graph(object):
    def __init__(self):
        super(Graph, self).__init__()
        self.feature_nodes = set()
        self.label_nodes = set()
        # TODO maintain self.root_node

    def add_feature(self, node):
        self.feature_nodes.add(node)

    def add_label(self, node):
        self.labels_nodes.add(node)

    def to_description(self, root_node, **kw):
        return root_node.to_description()



# importing at the end of the file to work around circular imports
from cntk.ops import *
