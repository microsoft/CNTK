import os
import numpy as np

_FLOATX = 'float32'
if "CNTK_EXECUTABLE_PATH" not in os.environ:
    raise ValueError("you need to point environmental variable 'CNTK_EXECUTABLE_PATH' to the CNTK binary")

CNTK_EXECUTABLE_PATH = os.environ['CNTK_EXECUTABLE_PATH']
CNTK_TRAIN_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_train_template.cntk")
CNTK_PREDICT_TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "cntk_predict_template.cntk")
CNTK_TRAIN_CONFIG_FILENAME = "train.cntk"
CNTK_PREDICT_CONFIG_FILENAME = "predict.cntk"
CNTK_OUTPUT_FILENAME="out.txt"

try:
    import pydot_ng as pydot
    PYDOT = True
except:
    try: 
        import pydot
        PYDOT = True
    except:
        PYDOT = False

class Context(object):
    import cntk.graph as graph
    def __init__(self, desc):
        
        self.directory = os.path.abspath('_cntk_%s'%id(desc))
        if os.path.exists(self.directory):
            print("Directory '%s' already exists - overwriting data."%self.directory) 
        else:
            os.mkdir(self.directory)

        ''' TODO: re-factor
        self['ModelDescription'] = desc
        self['ModelPath'] = os.path.join(self.directory, 'Model', 'model.dnn')
        self["FeatureDimension"] = self.X.shape[1]
        self["LabelDimension"] = self.y.shape[1]
        self['LabelType'] = "category" # TODO

               
        self.label_node = graph.Input(self.y.shape, var_name='labels')
        unique_labels = np.unique(self.y)

        crit_node_name, output_node_name, model_desc = self._gen_model_description()
        
        self['CriteriaNodes'] = crit_node_name
        self['EvalNodes'] = "DUMMY"
        self['OutputNodes'] = output_node_name
        self['TrainFile'] = self._get_train_file()
        self['LabelMappingFile'] = self._get_label_mapping_file(unique_labels)        
        self['NumOfClasses'] = len(unique_labels)

        # SGD
        self['MinibatchSize'] = batch_size
        self['LearningRate'] = self.model.optimizer.lr.get_value()
        self['MaxEpochs'] = nb_epoch
        
        
        #Predict
        self.X, self.y = input_data       
        self.y = np.expand_dims(self.y, 1)        
        self['PredictInputFile'] = self._get_test_file()
        self['PredictOutputFile'] = self._get_output_file()        
        self['LabelMappingFile'] = self._get_label_mapping_file()                    
        
        ''' 

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        pass

    #TODO: refactor those methods
    def _get_train_file(self):        
        data = np.hstack([self.X, self.y])
        filename = os.path.join(self.context.directory, 'input.txt')
        format_str = ' '.join(['%f']*self.X.shape[1] + ['%i'])
        np.savetxt(filename, data, delimiter=' ', newline='\r\n',
                fmt=format_str)
        return filename

    def _get_label_mapping_file(self, unique_labels):
        filename = os.path.join(self.context.directory, 'labelMap.txt')
        np.savetxt(filename, unique_labels, delimiter=' ', newline='\r\n', fmt='%i')
        return filename

    def _name(self, x):
        if hasattr(x, 'name'):
            name = x.name
        else:
            name = '?'

        if hasattr(x, 'get_shape'):
            name += ' %s'%str(x.get_shape())

        return name

    def _unroll_node(self, output, desc):
        param_variable_names = []
        if output.params:
            for p in output.params:
                if hasattr(p, 'eval') and p.name:
                    child_var, child_desc = self._unroll_node(p, desc)
                    param_variable_names.append(child_var)


        var_name = output.var_name or "v%i"%self.node_counter 
        self.node_counter+=1
        
        params = output.get_cntk_param_string(param_variable_names)

        line = "%s = %s(%s)"%(var_name, output.name, params)
        desc.append((var_name, line))

        return var_name, desc

    def _gen_model_description(self):
        # TODO layer.class_mode should determine the eval node 
        # ('categorical' or 'binary')

        self.node_counter = 0

        computation_root_node = self.model.get_output()

        # append the loss/eval node
        if self.model.loss.__name__=='categorical_crossentropy':
            # TODO This is a short-term hack just to get it running.
            # Correctly, we would need to execute the loss function on
            # placeholders to then get back the evaluation graph, which we then
            # would unroll into CNTK config. Unfortunately, many of the
            # operators are not supported, which is why we test for the
            # function name for now.
            eval_node = graph.Operator("CrossEntropy", (self.label_node, computation_root_node), 
                    get_output_shape=lambda x,y: x.get_shape())
        else:
            raise NotImplementedError

        _, log = self._unroll_node(eval_node, [])


        criteria_node_name = log[-1][0]
        output_node_name = log[-2][0]

        return criteria_node_name, output_node_name, "\r\n".join(line for var_name, line in log)
    

    def execute(self):
        filename = os.path.join(self.context.directory, CNTK_TRAIN_CONFIG_FILENAME)
        tmpl = open(CNTK_TRAIN_TEMPLATE_PATH, "r").read()
        with open(os.path.join(self.context.directory, filename), "w") as out:
            cntk_config_content = tmpl%self
            out.write(cntk_config_content)
            
        import subprocess
        subprocess.check_call([CNTK_EXECUTABLE_PATH, "configFile=%s"%filename])

        print("Wrote to directory %s"%self.context.directory)

        if PYDOT:
            # create a node graph that can be viewed with GraphViz or at http://sandbox.kidstrythisathome.com/erdos/
            g=pydot.Dot()
            self.write_pydot(g, self.model.get_output())
            g.write_raw(os.path.join(self.context.directory, "graph.dot"))

    def write_pydot(self, g, output, node_counter=0):
        var_name = "v%i"%node_counter 
        node_counter+=1

        param_nodes = []
        if output.params:
            for p in output.params:
                if hasattr(p, 'eval') and p.name:
                    param_nodes.append(self.write_pydot(g, p))

        node_name = self._name(output)
        node = pydot.Node(node_name)
        g.add_node(node)
        for var_child, child in param_nodes:
            g.add_edge(pydot.Edge(child, node))

        return var_name, node
    
    def _get_test_file(self):                        
        data = np.hstack([self.X, self.y])
        filename = os.path.join(self.context.directory, 'test.txt')
        format_str = ' '.join(['%f']*self.X.shape[1] + ['%i'])
        np.savetxt(filename, data, delimiter=' ', newline='\r\n',
                fmt=format_str)
        return filename    
        
    def _get_output_file(self):
        return os.path.join(self.context.directory, CNTK_OUTPUT_FILENAME)
    
    ''' TODO: re-implement predict as other actions as part of the context
    def execute(self):
        config_filename = os.path.join(self.context.directory, CNTK_PREDICT_CONFIG_FILENAME)
        tmpl = open(CNTK_PREDICT_TEMPLATE_PATH, "r").read()
        config_file_path=os.path.join(self.context.directory, config_filename)
        with open(config_file_path, "w") as out:
            cntk_config_content = tmpl%self
            out.write(cntk_config_content)
            print("Wrote to directory %s"%self.context.directory)
            
        import subprocess
        subprocess.check_call([CNTK_EXECUTABLE_PATH, "configFile=%s"%config_filename])

        # We get one out.txt.<node> file per output node. CNTK supports the
        # output of multiple output nodes. We support only one here.
        import glob
        out_file_wildcard = os.path.join(self.context.directory, CNTK_OUTPUT_FILENAME+'.*')
        out_filenames = glob.glob(out_file_wildcard)
        if len(out_filenames)!=1:
            raise ValueError('expected exactly one file starting with "%s", but got %s'%(CNTK_OUTPUT_FILENAME, out_filenames))

        data = np.loadtxt(out_filenames[0])

        return [data]
    '''
    
def fit(model, ins, batch_size, np_epoch):
    #import ipdb;ipdb.set_trace()
    with Context(model) as cm:        
        cm.execute()

def predict(model, ins, verbose=0):
    with Context(model) as cm:        
        cm.execute()
    

