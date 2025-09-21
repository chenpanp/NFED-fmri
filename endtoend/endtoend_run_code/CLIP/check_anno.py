import pickle

with open(r'G:\zwc\2023Challenge\CLIP\anno\subj01_trn_anno.pickle', 'rb') as f:
    data = pickle.load(f)

    # for index, element in enumerate(data):
    #     print(f"索引 {index}: {element}")


    if 13 in data:
        print('有有有有有')
    else:
        print('无')

    num_elements = len(data)
    nsd_id = 13
    data = data[nsd_id]


# print(data)

