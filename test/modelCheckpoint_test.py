import os

def test():
    print("running test function")

path_to_save_model = './output/saved_models'
if not os.path.isdir(path_to_save_model):
    os.makedirs(path_to_save_model)

# ModelCheckpoint callback saves a model at some interval. 
fileName = "weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5"   # File name includes epoch and validation accuracy.
path_to_save_model = os.path.join(path_to_save_model, fileName)
print(f"path_to_save_model: {path_to_save_model}")

print(test.__name__)