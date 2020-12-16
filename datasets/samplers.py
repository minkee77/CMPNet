import torch
import numpy as np


class CategoriesSampler():

    def __init__(self, label, n_batch, n_cls, n_per, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)

class CategoriesSampler_Semi():

    def __init__(self, label, n_batch, n_cls, n_shot, n_unlabel, n_query, ep_per_batch=1):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_shot = n_shot
        self.n_unlabel = n_unlabel
        self.n_query = n_query
        self.n_per = self.n_shot + self.n_unlabel + self.n_query
        self.ep_per_batch = ep_per_batch

        label = np.array(label)
        self.catlocs = []
        for c in range(max(label) + 1):
            self.catlocs.append(np.argwhere(label == c).reshape(-1))

    def __len__(self):
        return self.n_batch
    
    def __iter__(self):
        for i_batch in range(self.n_batch):
            batch = []
            for i_ep in range(self.ep_per_batch):
                episode = []
                classes = np.random.choice(len(self.catlocs), self.n_cls,
                                           replace=False)
                for c in classes:
                    l = np.random.choice(self.catlocs[c], self.n_per,
                                         replace=False)
                    episode.append(torch.from_numpy(l))
                episode = torch.stack(episode)
                batch.append(episode)
            batch = torch.stack(batch) # bs * n_cls * n_per
            yield batch.view(-1)


# class CategoriesSampler():

#     def __init__(self, label, n_batch, n_cls, n_shot, n_unlabel, n_query, ep_per_batch=1):
#         self.n_batch = n_batch
#         self.n_cls = n_cls
#         self.n_shot = n_shot
#         self.n_unlabel = int(n_unlabel/2)
#         self.n_query = n_query
#         self.n_per = self.n_shot + self.n_unlabel + self.n_query
#         self.ep_per_batch = ep_per_batch

#         label = np.array(label)
#         self.catlocs = []
#         for c in range(max(label) + 1):
#             self.catlocs.append(np.argwhere(label == c).reshape(-1))

#     def __len__(self):
#         return self.n_batch
    
#     def __iter__(self):
#         for i_batch in range(self.n_batch):
#             batch = []
#             for i_ep in range(self.ep_per_batch):
#                 episode = []
#                 classes_all = np.random.choice(len(self.catlocs), self.n_cls * 2,
#                                            replace=False)
#                 for c in classes_all[:self.n_cls]:
#                     l = np.random.choice(self.catlocs[c], self.n_per,
#                                          replace=False)

#                     for d in classes_all[self.n_cls:]:
#                         ll = np.random.choice(self.catlocs[d], int(self.n_unlabel/self.n_cls),
#                                          replace=False)
#                         l = np.insert(l, self.n_shot+self.n_unlabel, ll)
#                     episode.append(torch.from_numpy(l))

#                 episode = torch.stack(episode)
#                 batch.append(episode)
#             batch = torch.stack(batch) # bs * n_cls * n_per
#             yield batch.view(-1)
