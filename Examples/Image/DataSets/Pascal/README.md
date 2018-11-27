# Pascal VOC Dataset

The [Pascal VOC](http://host.robots.ox.ac.uk/pascal/VOC/) (PASCAL Visual Object Classes) data 
is a well known set of standardised images for object class recognition. 

## Getting the Pascal VOC data

The Pascal VOC dataset is not included in the CNTK distribution but can be easily
downloaded by running the following Python command:

`python install_pascalvoc.py`

This will download roughly 3.15GB of data and unpack it into the folder structure that is assumed in the [object recognition tutorial](https://docs.microsoft.com/en-us/cognitive-toolkit/Object-Detection-using-Fast-R-CNN#run-pascal-voc)

## Alternative: download data manually

You need the 2007 (trainval and test) and 2012 (trainval) data as well as the precomputed ROIs used in the original Fast R-CNN paper. 
For the object recognition tutorial you need to follow the folder structure described below. 

* Download and unpack the 2012 trainval data to `DataSets/Pascal/VOCdevkit`
  * Website: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/)
  * Devkit: [http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar)
* Download and unpack the 2007 trainval data to `DataSets/Pascal/VOCdevkit`
  * Website: [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/)
  * Devkit: [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar)
* Download and unpack the 2007 test data into the same folder `DataSets/Pascal/VOCdevkit`
  * [http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar)
* Download and unpack the precomputed ROIs to `DataSets/Pascal/selective_search_data`
  * [http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz](http://www.cs.berkeley.edu/~rbg/fast-rcnn-data/selective_search_data.tgz)

The `VOCdevkit/VOC2007` folder should contain at least the following (similar for 2012):
```
VOCdevkit/VOC2007
VOCdevkit/VOC2007/Annotations
VOCdevkit/VOC2007/ImageSets
VOCdevkit/VOC2007/JPEGImages
```

## Performance

If you are curious about how well computers can perform on Pascal VOC today, the official leaderboards are maintained at 
[http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php](http://host.robots.ox.ac.uk:8080/leaderboard/main_bootstrap.php)
