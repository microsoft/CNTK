
Logistic Regression
======
The Logistic Regression tool is a parallel implementation of the logistic regression on top of multiverso. It is a easy-to-use tool for training model on big data with numbers of machines. 

We test the tool in a Bing Ads click prediction dataset in Microsoft. The dataset is about 4TB with more than 5 billions of samples. The experiment is running on a cluster with 24 machines. Each machine has 20 physical cores and 256 GB ram and machines are connected with InfiniBand. The training of one epoch can be finished in about 18 minutes.


For more details, please refer to [wiki](https://github.com/Microsoft/multiverso/wiki/Logistic-Regression).
