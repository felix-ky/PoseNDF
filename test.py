import pickle
path = "/media/xky/_data/project/posendf/pickle_data/amass.pkl"
with open(path, 'rb') as file:
    dict = pickle.load(file)
print(len(dict['beta']))

print(dict['pose'][10000].shape)