
class ComputationNode(object):
    def __init__(self, name, params=None):
        self.name = name
        self.params = params
        self.var_name = None

    def __add__(self, other):
        return Plus(self, other)

    def __radd__(self, other):
        return Plus(other, self)

    def __matmul__(self, other):
        # NOTE supported in Python 3.5
        return Times(self, other)

    def __rmatmul__(self, other):
        # NOTE supported in Python 3.5
        return Times(other, self)

    # TODO more __operators__

    def _get_cntk_param_string(self, param_variable_names=None):
        params = ", ".join(param_variable_names)
        return params

    def eval(self, **kw):
        graph.eval(self, kw)

    def __str__(self):
        return "%s / params=%s"%(self.name, self.params)

    def _to_description(self, desc, unrolled_nodes, node_counter=0):
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
                        child_var, node_counter, child_desc = p_value._to_description(desc, unrolled_nodes, node_counter)
                        unrolled_nodes[p_value] = child_var
                    param_variable_names.append(child_var)
                else:
                    if isinstance(p_value, bool):
                        p_value = str(p_value).lower()
                    elif isinstance(p_value, str):
                        p_value = "'%s'"%p_value
                    
                    param_variable_names.append('%s=%s'%(p_name, p_value))

        var_name = self.var_name or "v%i"%node_counter 
        node_counter += 1
        
        params = self._get_cntk_param_string(param_variable_names)

        line = "%s = %s(%s)"%(var_name, self.name, params)
        desc.append(line)

        return var_name, node_counter, desc

    def to_description(self):
        unrolled_nodes = {}
        var_name, node_counter, desc = self._to_description(desc=[], unrolled_nodes=unrolled_nodes)
        return "\n".join(desc)



class Label(ComputationNode):
    def __init__(self, dims):
        super(Label, self).__init__('Input', params=('dims', 'tag'))
        self.dims = dims
        self.tag = 'label'

class Graph(object):
    def __init__(self):
        super(Graph, self).__init__()
        # TODO maintain self.root_node

    def to_description(self, root_node, **kw):
        return root_node.to_description()

    def eval(self, root_node, **kw):
        if root_node is None:
            root_node = self.root_node
        model_description = self.to_description(root_node)
        # TODO pull in context for config file generation, etc.


# importing at the end of the file to work around circular imports
from cntk.ops import *
