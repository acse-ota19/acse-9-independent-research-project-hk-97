from dataloader import DataPreparation
import numpy as np
from utils import *
from gan import GAN
from trainer import Trainer


features = ["userAcceleration.x", "userAcceleration.y", "userAcceleration.z"]#, "rotationRate.x", "rotationRate.y", "rotationRate.z"]
# act_labels = ["dws","ups"]#,"wlk", "jog"]#, "sit", "std"]
act_labels = ["dws"]


Loader = DataPreparation()
train_ts, test_ts,num_features, num_act_labels = Loader.ts(features, act_labels)





train_data, act_train_labels= Loader.time_series_to_section(train_ts.copy(),
                                                             num_act_labels,
                                                             sliding_window_size=200,
                                                             step_size_of_sliding_window=10,
                                                             standardize = False,
                                                             normalize = True,
                                                             mode="Train")

test_data, act_test_labels = Loader.time_series_to_section(test_ts.copy(),
                                                            num_act_labels,
                                                            sliding_window_size=200,
                                                            step_size_of_sliding_window=10,
                                                            standardize = False,
                                                            normalize = True,
                                                            mode="Test")
train_data = np.expand_dims(train_data,axis=3)
test_data = np.expand_dims(test_data,axis=3)


print("--> Shape of Training Sections:", train_data.shape)
print("--> Shape of Test Sections:", test_data.shape)

length = 200
expt_name = "dws"

create_directories(expt_name)
gan_ = GAN(length)
trainer_ = Trainer(gan_, expt_name, length)
trainer_.train_gan(epochs=10000, batch_size=64, sample_interval=10, train_data=train_data)