import json

L = []
for i in range(10):
    for j in range(1,10):
        som = '{0:.2f}'.format(i+j)
        sub = '{0:.2f}'.format(i-j)
        div = '{0:.2f}'.format(i/j)
#         div = round(i/j, 2)
        mult = '{0:.2f}'.format(i*j)
        if som not in L:
            L.append(som)
        if sub not in L:
            L.append(sub)
        if mult not in L:
            L.append(mult)
        if div not in L:
            L.append(div)

labels_dict = {value: key for (key, value) in enumerate(sorted(L))}
labels_dict['{0:.2f}'.format(-99999999999999)] = len(labels_dict)

inv_labels_dict = {v: k for k, v in labels_dict.items()}


with open('deep_equation/src/deep_equation/labels_dict.json', 'w') as fp:
    json.dump(labels_dict, fp)
    
    
with open('deep_equation/src/deep_equation/inv_labels_dict.json', 'w') as fp:
    json.dump(inv_labels_dict, fp)