import numpy as np

def inner_prod(dic_1, dic_2):
    key_1= set(dic_1.keys())
    key_2= set(dic_2.keys())
    intersc = key_1.intersection(key_2)

    if intersc:
        numer=0
        denom_1, denom_2 = 0.0, 0.0
        for key in intersc:
            numer += dic_1[key]*dic_2[key]
            denom_1 += dic_1[key]*dic_1[key]
            denom_2 += dic_2[key] * dic_2[key]
        return numer/(denom_1*denom_2)**(0.5)
    return 0.0

def sparcify(x):
    non_zero_indices = np.nonzero(x[0])[0]
    non_zero_values = x[0][non_zero_indices]
    dic = dict(zip(map(str, non_zero_indices), map(float, non_zero_values)))
    return dic

def sparse_to_numpy(x):
    out = np.zeros(4096)
    for key in x:
        out[int(key)] = x[key]
    return out

def create_doc(image_path, x):
    dic = sparcify(x)
    doc = {'image_id': image_path, 'sparse_features': dic, 'random': float(np.random.rand())}
    return doc
