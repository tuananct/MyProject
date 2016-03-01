from data_testing import FinalTaskData
import numpy as np
x = FinalTaskData()
#x.line_count()
#print x.num_lines
#y = np.array(x.arrayCreate()).reshape(1,150528)
#print y.shape
#np.
#merge = np.array([]).reshape(1,150528)
#print merge.shape
#print np.concatenate(y, merge).shape
x.loadData()
d = x.data_train["data"]
print d.shape
print d[0][2]
d = d.reshape((len(d), 3, 224, 224))
print d.shape
print d[0][0][0][2]
