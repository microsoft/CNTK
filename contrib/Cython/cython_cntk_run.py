import cython_cntk
print(dir(cython_cntk))

import time
#time.sleep(30)
res = cython_cntk.test()

#import ipdb;ipdb.set_trace()

print(res)
