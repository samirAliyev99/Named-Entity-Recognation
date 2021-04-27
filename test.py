
with open('ypred_1st.csv') as inn:
    f1 = inn.read().split('\n')

with open('ypred_2nd.csv') as inn:
    f2 = inn.read().split('\n')


for i, x in enumerate(zip(f1, f2)):
    if x[0] != x[1]:
        print(i)
        print(x[0])
        print(x[1])
        print()

        l1 = list(map(float, x[0].split(',')))
        print(max(l1), l1.index(max(l1)))

        l2 = list(map(float, x[1].split(',')))
        print(max(l2), l2.index(max(l2)))