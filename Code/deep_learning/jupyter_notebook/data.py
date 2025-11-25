import pickle
path = r"D:\0_final_project\DeepEnsemble\dataset-source\embedding-dataset\code2vec\GodClass_code2vec_embeddings.pkl"

with open(path, 'rb') as file:
    data = pickle.load(file)

print(type(data))

print(data)