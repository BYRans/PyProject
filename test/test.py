import datetime
import numpy as np

starttime = datetime.datetime.now()

#当中是你的程序



a = np.array([[0.123,0.887],[0.8,0.21111111111111]])

print(np.repeat(a,4).reshape(-1,8))

# tmp = np.vstack((a,b))
# np.random.shuffle(tmp)
# print(tmp)



# i = np.array([11,22,11,44])
# # print(np.random.choice(a,size=2))
# # tmp = np.hstack((i,a))
# # print(tmp)
# a11 = a[np.where(i==44)]
#
# print(a11)
# print("++++++++++a+++++++++++++++++")
# print(a)
# a[a[:,0] == 1,0] = 100
#
# print("--------------a-------------")
#
# print(a)