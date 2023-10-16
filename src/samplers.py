import torch
import numpy as np


class ActivitySubsetRandomSampler(torch.utils.data.SubsetRandomSampler):
    def __init__(self, data_source, samples_per_activity, num_acts, only_continuous_acts=False):
        '''Random sampling based on samples per activity

        Note that if a sample/window has multiple activities, it will happen
        that more than "samples_per_activity" samples are available for an
        activity.

        Parameters
        ----------
        data_source (torch.utils.data.DataSet): The used dataset
        samples_per_activity (int or float): How many samples to
            extract per activity. Can be in number of percent
        num_acts (int): Number of activities in the dataset
        only_continuous_acts (bool, optional): Consider only windows with
            continuous activity without any interruption
        '''
        self.in_samples = (samples_per_activity>=1 and \
                           type(samples_per_activity)==int)
        self.in_percent = (samples_per_activity<=1.0 and \
                           samples_per_activity>0.0 and \
                           type(samples_per_activity)==float)
        assert self.in_samples or self.in_percent, \
                'Provide subsample_perc either as float (0.0,1.0] or int>0'
        self.num_acts = num_acts
        if self.in_percent and samples_per_activity==1.0:
            print('No subsampling...')
            indices = list(range(len(data_source)))
        elif self.in_percent:
            _in_perc = 100*samples_per_activity
            print(f'Percentage subsampling with {_in_perc}% per activity')
            indices = self._get_indices(data_source, samples_per_activity)
        elif self.in_samples:
            print(f'Subsampling with {samples_per_activity} sample(s) per activity')
            indices = self._get_indices(data_source, samples_per_activity,
                                        only_continuous=only_continuous_acts)
        super().__init__(indices=indices, generator=None)

    def _get_indices(self, data_source, samples_per_activity, only_continuous=False):
        act_index_dict = {}
        for i, (_, y) in enumerate(data_source):
            act_in_i = list(set(y.numpy()))
            if only_continuous and len(act_in_i)>1:
                continue
            for a in act_in_i:
                if a not in act_index_dict:
                    act_index_dict[a] = [i]
                else:
                    act_index_dict[a] += [i]
        if len(act_index_dict)!=self.num_acts:
            raise ValueError('Not all activities in sampler')
        act_index_dict_lens = {k: len(v) for k,v in act_index_dict.items()}
        chosen_a_indices = {}
        already_selected_indices = []
        for a in sorted(act_index_dict_lens, key=act_index_dict_lens.get):
            indices_for_a = [i for i in act_index_dict[a] if i not in already_selected_indices]
            if self.in_samples:
                chosen = list(np.random.choice(indices_for_a,
                                               samples_per_activity))
                chosen_a_indices[a] = chosen
                already_selected_indices.append(chosen)
            elif self.in_percent:
                amount = int(len(act_index_dict[a])*samples_per_activity)
                chosen = list(np.random.choice(indices_for_a, amount))
                chosen_a_indices[a] = chosen
                already_selected_indices.append(chosen)
            if len(chosen)==0:
                raise ValueError(f'0 Samples for activity: {a}, increase subsample_perc!')
        return list(np.array(list(chosen_a_indices.values())).flatten())
