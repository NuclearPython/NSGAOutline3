M = []

L = [0 for i in range(10)]

print L

M.append(L)

print M

L[0] +=1

L[7] +=1

print L

M.append(L)

print M
