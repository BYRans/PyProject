a = ['a', 'a', 'b', 'c', 'a', 'f', 'd', 'c', 'e', 'e', 'a']
dic = {'a': 1, 'e': 2, 'd': 4}
selected = [x for x in a if x in dic.keys()] # 找到a中属于[1,5)中的元素
print(selected)