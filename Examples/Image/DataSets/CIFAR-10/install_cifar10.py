from __future__ import print_function
import cifar_utils as ut

if __name__ == "__main__":
    trn, tst= ut.loadData('http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz')
    print ('Writing train text file...')
    ut.saveTxt(r'./Train_cntk_text.txt', trn)
    print ('Done.')
    print ('Writing test text file...')
    ut.saveTxt(r'./Test_cntk_text.txt', tst)
    print ('Done.')

    print ('Converting train data to png images...')
    ut.saveTrainImages(r'./Train_cntk_text.txt', 'train')
    print ('Done.')
    print ('Converting test data to png images...')
    ut.saveTestImages(r'./Test_cntk_text.txt', 'test')
    print ('Done.')
