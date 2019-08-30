import numpy as np

class DataPreparation:
  
    """
    Object for data extraction, training and test set preparation, and
    feature scaling handling.
    
    The get_ds_infos(), ts() and time_series_to_section() methods have been
    developed from functions in the MotionSense dataset GitHub repository.
    A link to the original code from which it was based is:
    https://github.com/mmalekzadeh/motion-sense/blob/master/codes/gen_paper_codes/1_MotionSense_Trial.ipynb
    
    
    ...


    Methods
    -------
        get_ds_infos:           Inputs: None
                                Loads a .csv file from the MotionSense dataset 
                                containing metadata for the data collection 
                                trials.
                        
         ts:                    Inputs: features, activities
                                'ts' is an abbreviation for timeseries
                                Takes the features and activities which want to 
                                be considered, and generates an array of size
                                len(features) * len(activities) containing data
                                that can be used for training models
                                
        time_series_to_section: Inputs: dataset, num_act_labels, 
                                        sliding_window_size, 
                                        step_size_of_sliding_window, 
                                        standardize,
                                        normalize, 
                                        mode)
                                        
                                Slices array into smaller sections, creating
                                a training set.
                                
        obtain_std_nrm_paras:   Inputs: data_array
                                Abbreviation for obtain standardisation and
                                normalisation parameters.
                                Calculates the mean, std, min and max of data 
                                array along the axis containing the number of features
                                
        standardise:            Inputs: data_array
                                Performs data standardisation ( scaling such 
                                that the scaled data has mean=0 and std=1)
                                along each feature axis.
                                
        normalise:              Inputs: data_array
                                Performs data normalisation ( scaling such 
                                that the scaled data has max=1 and min=1)
                                along each feature axis.
                                
        un_normalise:           Inputs: data_array
                                Reverses the effect of normalise() using only
                                the statistics of the training set as calculated
                                using obtain_std_nrm_pars()
                                
        un_standardise:         Inputs: data_array
                                Reverses the effect of normalise() using only
                                the statistics of the training set as calculated
                                using obtain_std_nrm_pars()

    """

    def __init__(self):
      pass

    def get_ds_infos(self):
      """
      Loads metadata from 'data_subjects_info.csv'
      """

      ## 0:Code, 1:Weight, 2:Height, 3:Age, 4:Gender
      dss = np.genfromtxt("data_subjects_info.csv",delimiter=',')
      dss = dss[1:]
      print("----> Data subjects information is imported.")
      return dss

    def ts(self, features, activities):

        """ 

        'ts' is an abbreviation for timeseries. Takes the features and activities 
        which want to be considered, and generates an array of size len(features) 
        * len(activities) containing data that can be used for training models
        Parameters
        ----------
            features: (list) 
                      A list containing the string codes for the features under 
                      consideration.
                      e.g. features = ["userAcceleration.x", "userAcceleration.y"]
            activities: (list) 
                      A list containing the string codes for the activities under 
                      consideration.
                      e.g. features = ["dws", "ups", "walk"]

        Returns
        ---------
        train_data , test_data, num_features, num_act_labels
        """

        num_features = len(features)
        num_act_labels = len(activities)
        dataset_columns = num_features+num_act_labels

        label_codes = {"dws":num_features, "ups":num_features+1, "wlk":num_features+2, "jog":num_features+3, "sit":num_features+4, "std":num_features+5}
        trial_codes = {"dws":[1,2,11], "ups":[3,4,12], "wlk":[7,8,15], "jog":[9,16], "sit":[5, 13], "std":[6,14]}    

        new = {}

        for requested in trial_codes:
          if requested in activities:
            new[requested] = trial_codes[requested]

        trial_codes = new
        label_codes = {}
        count = 0
        for key in trial_codes:
          print(key, trial_codes[key])
          label_codes[key] = num_features + count
          count +=1

        print(label_codes)
        print(trial_codes)
        ds_list = self.get_ds_infos()
        train_data = np.zeros((0,dataset_columns))
        test_data = np.zeros((0,dataset_columns))
        for i, sub_id in enumerate(ds_list[:,0]):
            for j, act in enumerate(trial_codes):
                for trial in trial_codes[act]:
                    fname = 'A_DeviceMotion_data/'+act+'_'+str(trial)+'/sub_'+str(int(sub_id))+'.csv'
                    print("Loading: ", fname)
                    raw_data = pd.read_csv(fname)
                    raw_data = raw_data.drop(['Unnamed: 0'], axis=1)
                    raw_data = raw_data[features]
                    unlabel_data = raw_data.values

                    label_data = np.zeros((len(unlabel_data), dataset_columns))
                    label_data[:,:-(num_act_labels )] = unlabel_data
                    label_data[:,label_codes[act]] = 1
                    if trial > 10:
                        test_data = np.append(test_data, label_data, axis = 0)
                    else:    
                        train_data = np.append(train_data, label_data, axis = 0)

        return train_data , test_data, num_features, num_act_labels

    def time_series_to_section(self, dataset, num_act_labels, sliding_window_size, step_size_of_sliding_window, standardize = True, normalize = False, mode="Test"):
        """ 
        The dataset generated by ts() is sliced into smaller windows to generate a training set.
        This is returned alongside an array of 'one-hot' embedded vectors,containing the label which
        corresponds to the activity carried out during each window.

        Parameters
        ----------
            dataset:                      (array)    
                                          Dataset generated by the ts() function
                                          This is seperated into activities but
                                          not yet sliced into smaller windows
            num_act_labels:               (int)
                                          The number of activities being considered
                                          This is also return from ts()
            sliding_window_size:          (int)
                                          The size of the interval in number of
                                          timesteps. A value of 200 is recommended
                                          as this corresponds to 4 seconds
            step_size_of_sliding_window:  (int)
                                          Determines stride of sliding window as it
                                          slices the dataset
            standardize:                  (bool)
                                          Will standardise the dataset if True
            normalize:                    (bool)
                                          Will normalise the dataset if True
            mode:                         (str)
                                          Only if the mode=="Test", the parameters
                                          for the feature scaling will be stored

        Returns
        --------

        """

        data = dataset[: , 0:-(num_act_labels)]
        act_labels = dataset[: , -(num_act_labels):]


        if mode == "Train":
          self.obtain_std_nrm_paras(data)
        standardised_data = self.standardise(data)
        normalised_data = self.normalise(data)


        if normalize:

            data = normalised_data
            print("data has been normalised")

        elif standardize:
            data = standardised_data
            print("data has been standardised")
        else:
            print("----> Without Standardisation or Normalisation.....")

        ## We want the Rows of matrices show each Feature and the Columns show time points.
        data = data.T

        size_features = data.shape[0]
        size_data = data.shape[1]
        number_of_secs = round(((size_data - sliding_window_size)/step_size_of_sliding_window))

        ##  Create a 3D matrix for Storing Snapshots  
        secs_data = np.zeros((number_of_secs , size_features , sliding_window_size ))
        act_secs_labels = np.zeros((number_of_secs, num_act_labels))

        k=0    
        for i in range(0 ,(size_data)-sliding_window_size  , step_size_of_sliding_window):
            j = i // step_size_of_sliding_window
            if(j>=number_of_secs):
                break
            if(not (act_labels[i] == act_labels[i+sliding_window_size-1]).all()): 
                continue    
            secs_data[k] = data[0:size_features, i:i+sliding_window_size]
            act_secs_labels[k] = act_labels[i].astype(int)
            k = k+1
        secs_data = secs_data[0:k]
        act_secs_labels = act_secs_labels[0:k]

        return secs_data, act_secs_labels


    def obtain_std_nrm_paras(self, data):

      """
      Abbreviation for obtain standardisation and normalisation parameters. 
      Calculates the mean, std, min and max of data array along the axis 
      containing the number of features. It stores these values as
      attributes of the class so that they can be used to reverse the effects
      of normalisation and standardisation
      """

      means, stds = [], []
      mins, maxs = [], []

      for i in range(data.shape[1]):
        min_, max_ = np.min(data[:,i]), np.max(data[:,i])
        mean, std = np.mean(data[:,i], dtype=np.float64), np.std(data[:,i], dtype=np.float64)

        mins.append(min_)
        maxs.append(max_)

        means.append(mean)
        stds.append(std)

        self.standardise_params = [means, stds]
        self.normalise_params = [mins, maxs]


    def normalise(self,data):
      """
      Performs data normalisation (scaling such that the scaled data has max=1 
      and min=1) along each feature axis.
      """
      mins, maxs = self.normalise_params

      normalised = np.zeros((data.shape))
      for i in range(data.shape[1]):
        print(i)
        normalised[:,i] = (2 * (data[:,i] - mins[i])/(maxs[i] - mins[i])) -1


      return normalised

    def standardise(self,data):
      """
      Performs data standardisation ( scaling such that the scaled data has 
      mean=0 and std=1) along each feature axis. 
      """

      means, stds = self.standardise_params

      standardised = np.zeros((data.shape))
      for i in range(data.shape[1]):
        standardised[:,i] = (data[:,i] - means[i])/stds[i]

      return standardised


    def un_normalise(self, normalised_data):
      """
      Reverses the effect of normalise() using only the statistics of the training 
      set as calculated using obtain_std_nrm_pars()
      """
      mins, maxs = self.normalise_params
      print(mins,maxs)
      unnormalised = np.zeros((normalised_data.shape))

      for i in range(normalised_data.shape[1]):
        unnormalised[:,i] = (normalised_data[:,i] +1)/2 * (maxs[i] - mins[i]) + mins[i]

      return unnormalised


    def un_standardise(self, standardised_data):
      """
      Reverses the effect of standardise() using only the statistics of the training 
      set as calculated using obtain_std_nrm_pars()
      """
      means, stds = self.standardise_params

      unstandardised = np.zeros((standardised_data.shape))
      print(unstandardised.shape)
      for i in range(standardised_data.shape[1]):
        unstandardised[:,i] = (standardised_data[:,i] * stds[i]) +  means[i]

      return unstandardised

      
  
