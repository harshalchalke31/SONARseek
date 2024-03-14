import pandas as pd
import numpy as np
from numpy.random import normal

class DataDescription:
    def __init__(self, description_file,data_file):
        self.data_description = description_file
        self.class_data = data_file

    def get_std(self):
        stds=self.data_description.loc['std'].values
        return stds
    
    def get_class_means(self):
        return self.class_data.reset_index(drop=True).values

class DataGenerator:
    def __init__(self,data_description,class_data,n_samples=10000):
        self.description = DataDescription(data_description,class_data)
        self.n_samples = n_samples
        self.generated_data = pd.DataFrame()
    
    def generate(self):
        stds = self.description.get_std()
        class_means = self.description.get_class_means()
        samples_per_class = self.n_samples//2
        features = self.description.class_data.columns.tolist()

        generated_data=[]
        for index,means in enumerate(class_means):
            class_label = self.description.class_data.index[index]
            for _ in    range(samples_per_class):
                generated_row = np.clip(normal(loc=means,scale=stds),a_min=0,a_max=1)
                generated_data.append(np.append(generated_row,class_label))
        self.generated_data = pd.DataFrame(generated_data)
        return self.generated_data


