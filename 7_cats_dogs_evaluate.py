from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow import keras

base_dir = "/home/mpkuse/Downloads/kagglecatsanddogs_5340/PetImages/kaggle/subset"
test_dataset = image_dataset_from_directory( base_dir+"/test", 
                                             labels="inferred", 
                                             image_size=(180,180), 
                                             batch_size=32 )

model_file = '7_log_tensorboard/convnet_from_scratch.keras'
model_file = '7_log_with_data_aug_tensorboard/convnet_from_scratch.keras'
model_file = '7_log_with_data_aug_on_10k_trainimg_tensorboard/convnet_from_scratch.keras'
model_file = '7_log_vgg_finetune_on_subset_tensorboard/convnet_from_scratch.keras'

test_model = keras.models.load_model( model_file )
test_loss, test_acc = test_model.evaluate( test_dataset )
print(  'test_acc=', test_acc )