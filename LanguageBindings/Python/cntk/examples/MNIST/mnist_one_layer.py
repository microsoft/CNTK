#sys.path.insert(0, "E:\CNTK\LanguageBindings\Python")

from ..cntk import *

if (__name__ == "__main__"):
    
    # =====================================================================================
    # Softmax Regression with UCIFastReader
    # =====================================================================================
    
    # Network definition 
    
    x = Input(2)    
    y= Input(3, tag = 'label')    
    w = LearnableParameter(3,2)
    b = LearnableParameter(3,1) 
    t = Times(w,x)    
    out = Plus(t, b)
    out.tag = 'output'
    ec = CrossEntropyWithSoftmax(y, out)
    ec.tag = 'criterion'

    # Build the reader        
    r = UCIFastReader(filename="E:\CNTK\LanguageBindings\Python\Train-3Classes.txt", 
                      labels_node_name="v0", 
                      labels_dim=1, 
                      labels_start=2, 
                      num_of_classes=3, 
                      label_mapping_file="E:\CNTK\LanguageBindings\Python\SimpleMapping-3Classes.txt")
    
    # Add the input node to the reader
    r.add_input(x, "0", 2)
    
    # Build the optimizer
    my_sgd = SGD(epoch_size = 0, minibatch_size = 25, learning_ratesPerMB = 0.1, max_epochs = 3)
    
    # Create a context or re-use if already there
    with Context('demo1', optimizer= my_sgd, root_node= ec, clean_up=False) as ctx:            
        # CNTK actions
        ctx.train(r, False)        
        ctx.test(r)        
        ctx.predict(r)       
