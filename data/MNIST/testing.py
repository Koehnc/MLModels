import numpy as np

# lists = range(3,-1,-1)
# for i in lists:
#     print(i)


test = [[2,2],[2,2]]
answer = [[1,1],[1,1]]
choices = np.random.choice([False, True], size=[2,2])
print(choices)
test[choices] = answer[choices]

print(test)