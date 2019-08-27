from itertools import groupby
from collections import deque
# print([list(value) for _, value in groupby('aaabbbbeeeee')])

q = deque()
q.append(123)
if not q:
    print('Yes')
q.append('asd')
q.append(123)
print(q.pop())

