import math

def mean(values):
    return sum(values)/len(values)

def variance(values, m=None):
    if not m:
        m = mean(values)
    return sum([(v-m)**2 for v in values]) / len(values)

def prob(m, v, x):
    e = math.exp(-((x-m)**2)/(2*v))
    d = math.sqrt(2*math.pi*v)
    return e/d


values = [3, 12, 27, 14, 35, 35, 19, 65, 29, 31, 49, 56, 71, 20]
category = [1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0]
x = 26

# ---------------------------------------------------------------------------

assert len(values) == len(category), "Длина values и category не сопадает!"

cs = list(set(category))

for c in cs:
    values_c = [v for i, v in enumerate(values) if category[i] == c]
    m = mean(values_c)
    v = variance(values=values_c, m=m)
    p = prob(m=m, v=v, x=x)
    print("Class={}, mean={}, variance={}, P(x|c)={}".format(c, m, v, p), end="\n\n")

