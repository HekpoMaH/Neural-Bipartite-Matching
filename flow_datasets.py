import os
import re
import torch
from torch_geometric.data import Dataset, Data
import torch.utils.data as torch_data

def read_edge_index(filehandle):
    edge_index=[[], []]
    edge_index[0] = [int(x) for x in filehandle.readline().split()]
    edge_index[1] = [int(x) for x in filehandle.readline().split()]
    edge_index = torch.tensor(edge_index)
    return edge_index

def folder_elements(folder_name):
    if not os.path.isdir(folder_name):
        return 0
    return (len([_ for _ in os.listdir(folder_name)]))

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    def tryint(s):
        try:
            return int(s)
        except:
            return s
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

class GraphDatasetBase(Dataset):
    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None, less_wired=False, probp=1, probq=4):
        self.device = device
        self.less_wired = less_wired
        self.probp = probp
        self.probq = probq
        lw = '_less_wired' if self.less_wired else ''
        proot = ''
        if not(probp == 1 and probq == 4):
            proot = '_'+str(probp)+'_'+str(probq)
        root += lw
        root += proot
        print("ROOT", root)
        assert split in ['train', 'val'] or 'test' in split
        self.split = split
        # print("INIT FIN")
        super(GraphDatasetBase, self).__init__(root, transform, pre_transform)
        print("PROCESSED", self.processed_dir)

    @property
    def raw_file_names(self):
        dirname = self.raw_dir+'/'+self.split
        raw_file_names = [self.split+'/'+str(x)+'.txt' for x in range(1, folder_elements(dirname)+1)]
        return raw_file_names

    @property
    def processed_file_names(self):
        # Raw dir below is not a typo
        dirname = self.raw_dir + '/' + self.split
        processed_file_names = [self.split+'/'+str(x)+'.pt' for x in range(0, folder_elements(self.processed_dir+'/'+self.split))]
        return processed_file_names

    def __len__(self):
        if not os.path.isdir('./' + self.raw_dir+ '/' + self.split):
            self.download()
        if not os.path.isdir('./' + self.processed_dir+ '/' + self.split):
            self.process()
        return (len([_ for _ in os.listdir(self.processed_dir+'/'+self.split)]))

    def process_augmenting_iteration(self, file_handle, s, n):
            metadata = {}
            metadata['weights'] = torch.tensor([float(x) for x in file_handle.readline().split()])
            metadata['capacities'] = torch.tensor([float(x) for x in file_handle.readline().split()])

            edge_attr = torch.cat((metadata['weights'].view(-1, 1), metadata['capacities'].view(-1, 1)), dim=1)

            bf = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
            pred = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
            features = torch.cat((bf, pred), dim=1).unsqueeze(1)
            target_features = None

            for i in range(s+n+1):
                target_bf = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
                target_pred = torch.tensor([float(x) for x in file_handle.readline().split()]).unsqueeze(1)
                y = torch.cat((target_bf, target_pred), dim=1)
                if target_features is None:
                    target_features = y.unsqueeze(1)
                else:
                    target_features = torch.cat((target_features, y.unsqueeze(1)), dim=1)

                if i != s+n-1:
                    features = torch.cat((features, y.unsqueeze(1)), dim=1)
            return edge_attr, features, target_features, metadata

    def process_graph(self, file_handle):
        s, n = [int(x) for x in file_handle.readline().split()]
        edge_index=[[], []]
        edge_index[0] = [int(x) for x in file_handle.readline().split()]
        edge_index[1] = [int(x) for x in file_handle.readline().split()]
        edge_index = torch.tensor(edge_index)
        return s, n, edge_index

    def preload(self):
        self.preloaded_data = []
        for path in self.processed_paths:
            print(path)
            self.preloaded_data.append(torch.load(path).to(self.device))

    def download(self):
        if not os.path.isdir(self.root+'/raw'):
            if not self.less_wired:
                os.system('./gen.sh 8 8 all_iter')
            else:
                os.system('./gen.sh 8 8 all_iter_less_wired'+' '+str(self.probp)+' '+str(self.probq))

    def get(self, idx):
        if not os.path.isdir(os.path.join(self.processed_dir+'/'+self.split)):
            if not os.path.isdir(os.path.join(self.raw_dir+'/'+self.split)):
                self.download()
            self.process()
        
        data = torch.load(os.path.join(self.processed_dir+'/'+self.split, '{}.pt'.format(idx))).to(self.device)
        return data

class SingleIterationDataset(GraphDatasetBase):
    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        print("PRP", self.raw_paths)
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)

                while True:
                    edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                    if edge_attr.nelement() == 0:
                        break
                    metadata['reachability'] = target_features[:, :, 1] != -1

                    data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                    if not os.path.isdir(os.path.join(dirname)):
                        os.mkdir(os.path.join(dirname))
                    torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))

                    cnt += 1

class BFSSingleIterationDataset(GraphDatasetBase):

    def __init__(self, root, device='cuda', split='train', transform=None, pre_transform=None, less_wired=False, probp=1, probq=4):
        super(BFSSingleIterationDataset, self).__init__(root, device, split, transform, pre_transform, less_wired, probp, probq)


    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)

                while True:
                    edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                    if edge_attr.nelement() == 0:
                        break
                    edge_attr[:, 1] = (metadata["capacities"] > 0).float()

                    features = (features[:, :, 1] != -1).float()
                    target_features = (target_features[:, :, 1] != -1).float()

                    data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                    if not os.path.isdir(os.path.join(dirname)):
                        os.mkdir(os.path.join(dirname))
                    torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))

                    cnt += 1

class GraphOnlyDataset(GraphDatasetBase):

    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)
                edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                if not os.path.isdir(os.path.join(dirname)):
                    os.mkdir(os.path.join(dirname))
                torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))
                cnt += 1

class GraphOnlyDatasetBFS(GraphDatasetBase):

    def process(self):
        cnt = 0
        dirname = self.processed_dir+'/'+self.split
        for raw_path in sorted(self.raw_paths, key=alphanum_key):
            with open(raw_path, "r") as raw_file:
                s, n, edge_index = self.process_graph(raw_file)
                edge_attr, features, target_features, metadata = self.process_augmenting_iteration(raw_file, s, n)
                edge_attr[:, 1] = (metadata["capacities"] > 0).float()
                features = (features[:, :, 1] != -1).float()
                target_features = (target_features[:, :, 1] != -1).float()
                data = Data(features, edge_index, edge_attr, y=target_features, **metadata)
                if not os.path.isdir(os.path.join(dirname)):
                    os.mkdir(os.path.join(dirname))
                torch.save(data, os.path.join(dirname, '{}.pt'.format(cnt)))
                cnt += 1

if __name__ == '__main__':
    f = SingleIterationDataset('./all_iter', split='test', less_wired=True, device='cpu', probp=3, probq=4)
    print(f[0].x[:, 0])
    print(f[0].capacities)
    print(f[0].num_nodes)
