Copy-pasteable command lines for debugging the Tests\EndToEndTests\Speech tests in VS
=====================================================================================

We provide the Visual Studio debug command arguments for the 'Examples' end-to-end tests below. You can always generate the Visual Studio debug command arguments using the  -n  options on the TestDriver for a specific end-to-end test:

 python TestDriver.py run -n Image/QuickE2E 

(For more details on using the TestDriver see https://docs.microsoft.com/en-us/cognitive-toolkit/How-to-Test#end-to-end-tests ). From the output of the above command you simply copy the 'VS debugging command args' to the command arguments of the CNTK project in Visual Studio (Right click on CNTK project -> Properties -> Configuration Properties -> Debugging -> Command Arguments). Start debugging the CNTK project.

Note: Below, the current dir is sometimes set to be the data directory. This allows for local paths in data scripts.

Note: To redirect the log output you can optionally add   stderr=$(SolutionDir)Tests\EndToEndTests\myLogFileName.log

--- Examples/Image/MNIST/01_OneHidden 

COMMAND:  configFile=$(SolutionDir)\Examples\Image\MNIST\Config/01_OneHidden.config currentDirectory=$(SolutionDir)\Tests\EndToEndTests\Image\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_01_OneHidden@debug_gpu DataDir=$(SolutionDir)\Tests\EndToEndTests\Image\Data ConfigDir=$(SolutionDir)\Examples\Image\MNIST\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_01_OneHidden@debug_gpu DeviceId=0 train=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Train.txt]] test=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Test.txt]] train=[SGD=[maxEpochs=1]] train=[SGD=[epochSize=100]] train=[reader=[randomize=none]] imageLayout="cudnn"

--- Examples/Image/MNIST/02_Convolution 

COMMAND:  configFile=$(SolutionDir)\Examples\Image\MNIST\Config/02_Convolution.config currentDirectory=$(SolutionDir)\Tests\EndToEndTests\Image\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_02_Convolution@debug_gpu DataDir=$(SolutionDir)\Tests\EndToEndTests\Image\Data ConfigDir=$(SolutionDir)\Examples\Image\MNIST\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_02_Convolution@debug_gpu DeviceId=0 train=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Train.txt]] test=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Test.txt]] train=[SGD=[maxEpochs=1]] train=[SGD=[epochSize=128]] train=[reader=[randomize=none]] imageLayout="cudnn"

--- Examples/Image/MNIST/03_ConvBatchNorm 

COMMAND:  configFile=$(SolutionDir)\Examples\Image\MNIST\Config/03_ConvBatchNorm.config currentDirectory=$(SolutionDir)\Tests\EndToEndTests\Image\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_03_ConvBatchNorm@debug_gpu DataDir=$(SolutionDir)\Tests\EndToEndTests\Image\Data ConfigDir=$(SolutionDir)\Examples\Image\MNIST\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Image\MNIST_03_ConvBatchNorm@debug_gpu DeviceId=0 train=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Train.txt]] test=[reader=[file=$(SolutionDir)\Tests\EndToEndTests\Image\Data/Test.txt]] train=[SGD=[maxEpochs=1]] train=[SGD=[epochSize=128]] train=[reader=[randomize=none]] imageLayout="cudnn"

--- Tests/EndToEndTests/Simple2d/MultiGpu 

COMMAND:  configFile=$(SolutionDir)\Tests\EndToEndTests\Simple2d/Multigpu.config currentDirectory=$(SolutionDir)\Tests\EndToEndTests\Simple2d\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Tests\EndToEndTests\Simple2d_MultiGpu@debug_gpu DataDir=$(SolutionDir)\Tests\EndToEndTests\Simple2d\Data ConfigDir=$(SolutionDir)\Tests\EndToEndTests\Simple2d OutputDir=$(SolutionDir)\Tests\EndToEndTests\Tests\EndToEndTests\Simple2d_MultiGpu@debug_gpu DeviceId=0 Multigpu_Demo_Train=[SGD=[maxEpochs=1]] Multigpu_Demo_Train=[SGD=[epochSize=100]] Multigpu_Demo_Train=[reader=[randomize=none]]

--- Tests/EndToEndTests/Simple2d/OneHidden 

COMMAND:  configFile=$(SolutionDir)\Tests\EndToEndTests\Simple2d/OneHidden.config currentDirectory=$(SolutionDir)\Tests\EndToEndTests\Simple2d\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Tests\EndToEndTests\Simple2d_Simple@debug_gpu DataDir=$(SolutionDir)\Tests\EndToEndTests\Simple2d\Data ConfigDir=$(SolutionDir)\Tests\EndToEndTests\Simple2d OutputDir=$(SolutionDir)\Tests\EndToEndTests\Tests\EndToEndTests\Simple2d_Simple@debug_gpu DeviceId=0 Simple_Demo_Train=[SGD=[maxEpochs=1]] Simple_Demo_Train=[SGD=[epochSize=100]] Simple_Demo_Train=[reader=[randomize=none]]

--- Examples/Speech/AN4/FeedForward 

COMMAND:  configFile=$(SolutionDir)\Examples\Speech\AN4\Config/FeedForward.config currentDirectory=$(SolutionDir)\Examples\Speech\AN4\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Speech\AN4_FeedForward@debug_gpu DataDir=$(SolutionDir)\Examples\Speech\AN4\Data ConfigDir=$(SolutionDir)\Examples\Speech\AN4\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Speech\AN4_FeedForward@debug_gpu DeviceId=0 speechTrain=[SGD=[maxEpochs=1]] speechTrain=[SGD=[epochSize=2048]]

--- Examples/Speech/AN4/LSTM 

COMMAND:  configFile=$(SolutionDir)\Examples\Speech\AN4\Config/LSTM-NDL.config currentDirectory=$(SolutionDir)\Examples\Speech\AN4\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Speech\AN4_LSTM@debug_gpu DataDir=$(SolutionDir)\Examples\Speech\AN4\Data ConfigDir=$(SolutionDir)\Examples\Speech\AN4\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\Speech\AN4_LSTM@debug_gpu DeviceId=0 speechTrain=[SGD=[maxEpochs=1]] speechTrain=[SGD=[epochSize=64]] parallelTrain=false

--- Examples/SequenceToSequence/PennTreebank/RNN 

COMMAND:  configFile=$(SolutionDir)\Examples\SequenceToSequence\PennTreebank\Config/rnn.config currentDirectory=$(SolutionDir)\Examples\SequenceToSequence\PennTreebank\Data RunDir=$(SolutionDir)\Tests\EndToEndTests\Examples\SequenceToSequence\PennTreebank_RNN@debug_gpu DataDir=$(SolutionDir)\Examples\SequenceToSequence\PennTreebank\Data ConfigDir=$(SolutionDir)\Examples\SequenceToSequence\PennTreebank\Config OutputDir=$(SolutionDir)\Tests\EndToEndTests\Examples\SequenceToSequence\PennTreebank_RNN@debug_gpu DeviceId=0 train=[SGD=[maxEpochs=1]] train=[epochSize=2048]] trainFile=ptb.small.train.txt validFile=ptb.small.valid.txt testFile=ptb.small.test.txt confVocabSize=1000

