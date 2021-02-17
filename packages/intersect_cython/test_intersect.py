from pyintersect import intersect
import numpy as np

s1 = np.array([1,2,3,4,5,6,7,8,9,20], dtype=np.uint32)
s2 = np.array([4,5,6,7], dtype=np.uint32)

out = intersect(s1, s2)
print (out)






