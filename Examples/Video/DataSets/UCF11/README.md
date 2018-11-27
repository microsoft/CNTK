# UCF YouTube Action Data Set

The UCF11 dataset (http://crcv.ucf.edu/data/UCF_YouTube_Action.php) for action recognition is one of the most widely used video dataset for experimenting with different classification algorithms. UCF11 contains 1160 videos, each is labeled to one of the following 11 action categories: biking/cycling, diving, golf swinging, horse back riding, soccer juggling, swinging, tennis swinging, trampoline jumping, volleyball spiking, and walking with a dog.

## Setup

UCF11 dataset is not included in the CNTK distribution but can be easily be
downloaded and converted to CNTK-supported format. But first let's install our dependency, all examples, including the setup script, depend on `imageio` package, to install imageio do the following:

* For Anaconda: `conda install -c pyzo imageio`
* For pip: `pip install imageio`

Now we are ready to download and setup UCF11 dataset by running the following Python command:

`python install_ucf11.py`

After running the script, you will see two output files in the current folder: train_map.csv and test_map.csv. The total amount of disk space required is around `1`GB. You may now proceed to the [`GettingStarted`](../../GettingStarted) folder to play with this dataset.

If you already have download the dataset locally, you can simply run the following script to generate train_map.csv and test_map.csv.

`split_ucf11.py -i <unzipped location of the dataset> -o <UCF11 folder>`

## Reference
>Jingen Liu, Jiebo Luo and Mubarak Shah, Recognizing Realistic Actions from Videos "in the Wild", IEEE International Conference on Computer Vision and Pattern Recognition(CVPR), 2009.
