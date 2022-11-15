import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
#from xgboost import train
from torch_geometric.data import InMemoryDataset, download_url, Data, HeteroData, Dataset, extract_zip
import glob
import networkx as nx
import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import pathlib

column_dict = {'id': 'TX_ID', 'sender': 'User', 'receiver': 'Merchant Name', 'label': 'Is Fraud?'}

original_features = ['Amount', 'Insufficient Balance', 'Bad PIN',
       'Technical Glitch', 'Bad Card Number', 'Bad CVV', 'Bad Expiration',
       'Bad Zipcode', 'total_seconds',
       'sin_time_seconds', 'cos_time_seconds', 'sin_time_days',
       'cos_time_days', 
       'TX_CAT_1', 'TX_CAT_3', 'TX_CAT_4', 'TX_CAT_5',
       'TX_CAT_6', 'TX_CAT_7', 'TX_CAT_8', 'TX_CAT_9', 'TX_TYPE_0',
       'TX_TYPE_1', 'TX_TYPE_2', 'MC_CONTINENT_AF', 'MC_CONTINENT_AS',
       'MC_CONTINENT_EU', 'MC_CONTINENT_NA', 'MC_CONTINENT_OC',
       'MC_CONTINENT_SA', 'MC_CONTINENT_US'
       ]

urls = {5: "https://www.googleapis.com/drive/v3/files/1bVsEMaQhnMgeuyaijJJ6awI9xN_LfLUC?alt=media&key=AIzaSyBA5Am3Lv6YXC6iBgld_OGy9ChZDoyYm90"}

class FraudSubset(InMemoryDataset):
    def __init__(self, root, features=None, column_dict=None, features_requiring_scaling=None, file_name = 'data', weighted = False, transform=None, pre_transform=None, pre_filter=None):
        self.features = features
        self.column_dict = column_dict
        self.features_requiring_scaling = features_requiring_scaling
        self.file_name = file_name
        self.weighted = weighted
        super().__init__(root, transform, pre_transform, pre_filter)
        

    @property
    def raw_file_names(self):

        return glob.glob(os.path.join(self.root + '/*.feather'))

    @property
    def processed_file_names(self):
        return [(str(self.file_name) + '_{}.pt').format(str(i)) for i in range(10)]

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root, 'raw')




    def process(self):
        # Read data into huge `Data` list.

        test_files = sorted(glob.glob(os.path.join(self.raw_dir + '/*_test.feather')))
        train_files = sorted(glob.glob(os.path.join(self.raw_dir + '/*_train.feather')))
        
        idx = 0
        for train_file, test_file in tqdm(zip(train_files, test_files)):
            
            data = read_fraud_data(train_file, test_file, features=self.features, weighted=self.weighted, column_dict=self.column_dict, features_requiring_scaling=self.features_requiring_scaling)

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir, str(self.file_name)+f'_{idx}.pt'))
            idx += 1

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, str(self.file_name)+ f'_{idx}.pt'))
        return data



    

def read_fraud_data( train_file, test_file, features = [],  weighted = False, embeddings=False, features_requiring_scaling=None, column_dict = {}):

    # Read both files
    df_train = pd.read_feather(train_file)
    #df_train = df_train.set_index(df_train.columns[0])
    df_test = pd.read_feather(test_file)
    #df_test = df_test.set_index(df_test.columns[0])

    frac_val = 0.1
    cutoff = int(df_train.shape[0] * (1-frac_val))
    df_val = df_train.iloc[cutoff:]
    df_train = df_train.iloc[:cutoff]
    print("df_train_shape", str(df_train.shape))
    print("df_val_shape", str(df_val.shape))
    print("df_test_shape", str(df_test.shape))

    # 
    # Feature scaling
    if features_requiring_scaling:
        scaler = MinMaxScaler()
        scaler.fit(df_train.loc[:, features_requiring_scaling])
        df_train_scaled = pd.DataFrame(scaler.transform(df_train.loc[:, features_requiring_scaling]), columns=features_requiring_scaling, index=df_train.index)
        df_train = df_train.drop(features_requiring_scaling, axis=1)
        df_train = df_train.merge(df_train_scaled, left_index=True, right_index=True)
        df_val_scaled = pd.DataFrame(scaler.transform(df_val.loc[:, features_requiring_scaling]), columns=features_requiring_scaling, index=df_val.index)
        df_val = df_val.drop(features_requiring_scaling, axis=1)
        df_val = df_val.merge(df_val_scaled, left_index=True, right_index=True)
        df_test_scaled = pd.DataFrame(scaler.transform(df_test.loc[:, features_requiring_scaling]), columns=features_requiring_scaling, index=df_test.index)
        df_test = df_test.drop(features_requiring_scaling, axis=1)
        df_test = df_test.merge(df_test_scaled, left_index=True, right_index=True)

    print("df_train_shape", str(df_train.shape))
    print("df_val_shape", str(df_val.shape))
    print("df_test_shape", str(df_test.shape))

    df = pd.concat([df_train, df_val, df_test])

    print(df.shape)
    print(df.loc[:, column_dict['label']].value_counts())
    print(column_dict['id'])
    print(column_dict['sender'])
    print(column_dict['receiver'])
    print(column_dict['label'])
    
    G = nx.Graph()
    G.add_nodes_from(df.loc[:, column_dict['id']].unique(), type='TX')
    G.add_nodes_from(df.loc[:, column_dict['sender']].unique(), type='other')
    G.add_nodes_from(df.loc[:, column_dict['receiver']].unique(), type='other')

    if weighted:
        G.add_weighted_edges_from(zip(df.loc[:, column_dict['id']], df.loc[:, column_dict['sender']], df.loc[:, column_dict['edge_weight']]))
        G.add_weighted_edges_from(zip(df.loc[:, column_dict['id']], df.loc[:, column_dict['receiver']], df.loc[:, column_dict['edge_weight']]))
    else: 
        G.add_edges_from(zip(df.loc[:, column_dict['id']], df.loc[:, column_dict['sender']]))
        G.add_edges_from(zip(df.loc[:, column_dict['id']], df.loc[:, column_dict['receiver']]))

    #print('done networkx')

    # determines the order of nodes
    tx_list = list(df.loc[:, column_dict['id']].unique())
    card_list = list(df.loc[:, column_dict['sender']].unique())
    merchant_list = list(df.loc[:, column_dict['receiver']].unique())
    nodelist = tx_list + card_list + merchant_list
    number_of_tx = len(list(df.loc[:, column_dict['id']].unique()))
    df_tx_index = df.set_index(column_dict['id'])

    # which features can be used for training?
    # Create a list of features that can be used during training (i.e. drop fraud label column)
    #features = df.columns.drop(['CARD_PAN_ID', 'TERM_MIDUID', 'TX_FRAUD', 'TX_ID'])
    
    #features = original_features.copy()
    #print(len(features))
    #if rfm:
    #    features += rfm_features
    #    print(len(features))
    #if apate:
    #    features += apate_features
    #    print(len(features))
    
    print(len(features))

    # Get y tensor
    a = df_tx_index.loc[tx_list, column_dict['label']].values.astype(int)
    b = np.zeros(len(card_list))
    c = np.zeros(len(merchant_list))
    y = np.concatenate((a,b,c))
    y = torch.tensor(y, dtype=torch.long)
    
    #print('done y')
    # delta is a very small number that is summed with the edge weights. This is to avoid that the sum of all edge weights of a node equals zero. 
    # In the neighborsampler the sum of the weights is used to normalize the weights to probabilities. Hence, a sum equaling zero would result in NaN values. 
    if weighted:
        edge_weight_col = df.loc[:, column_dict['edge_weight']]
        delta = min(edge_weight_col[edge_weight_col > 0]) / 100
    
    # Get edge_index
    adj = nx.to_scipy_sparse_array(G).tocoo()
    row = torch.from_numpy(adj.row.astype(np.int64)).to(torch.long)
    col = torch.from_numpy(adj.col.astype(np.int64)).to(torch.long)
    cell_data = torch.from_numpy(adj.data.astype(np.double)).to(torch.double)
    edge_index = torch.stack([row, col], dim=0)
    if weighted:
        edge_weight = cell_data + delta
    else:
        edge_weight = None
    #print('done edge_index')

    d = df_tx_index.loc[tx_list, features].values
    e = np.zeros((len(card_list), len(features)))
    f = np.zeros((len(merchant_list), len(features)))

    x = np.concatenate((d,e,f), axis=0)
    x = torch.tensor(x.astype(np.float), dtype=torch.float32)
    #print('done x')
    
    # Validation fraction from training data
    frac_val = 0.1
    train_mask = torch.zeros(y.size(0), dtype=torch.bool)
    val_mask = torch.zeros(y.size(0), dtype=torch.bool)
    test_mask = torch.zeros(y.size(0), dtype=torch.bool)

    train_cutoff = int(df_train.shape[0])
    val_cutoff = int(df_train.shape[0]) + int(df_val.shape[0])

    train_mask[:train_cutoff] = True
    val_mask[train_cutoff:val_cutoff] = True
    test_mask[val_cutoff:number_of_tx] = True

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, y=y, train_mask = train_mask, val_mask = val_mask, test_mask=test_mask)
    return data







class HeteroFraudSubset(InMemoryDataset):
    def __init__(self, root, subset, column_dict = {}, weighted=False, features_requiring_scaling = None, transform=None, pre_transform=None, pre_filter=None):
        self.column_dict = column_dict
        self.features_requiring_scaling = features_requiring_scaling
        self.weighted = weighted
        self.subset = subset
        self.url = urls[subset]
        root_subset = os.path.join(root, str(subset))
        super().__init__(root_subset, transform, pre_transform, pre_filter)
   
    @property
    def raw_file_names(self):

        return ['df'+str(self.subset).zfill(3)+'_test.feather', 'df'+str(self.subset).zfill(3)+'_train.feather']

    @property
    def processed_file_names(self):
        return ['subset'+str(self.subset) + '.pt']

    def download(self):
        path = download_url(self.url, self.raw_dir)
        extract_zip(path, self.raw_dir)
        os.unlink(path)


    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root,  'raw')

    def process(self):
        data_list = []
        # Read data into huge `Data` list.

        test_file = glob.glob(os.path.join(self.raw_dir + '/*_test.feather'))[0]
        train_file = glob.glob(os.path.join(self.raw_dir + '/*_train.feather'))[0]
        

        

        data = read_hetero_fraud_data(train_file, test_file, scaling=True, weighted=self.weighted)
    
        if self.pre_filter is not None and not self.pre_filter(data):
            pass

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, os.path.join(self.processed_dir, 'subset'+ str(self.subset)+'.pt'))



    def filter_TX_CH_edge(self, n1, n2):
        return self.G[n1][n2].get("type") == 'pays'

    def filter_TX_MC_edge(self, n1, n2):
        return self.G[n1][n2].get("type") == 'receives'


    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, 'subset'+str(self.subset) + '.pt'))
        return data

        


def read_hetero_fraud_data(train_file, test_file, scaling = True, weighted = False, frac_val = 0.1):

        

        # Read both files
        df_train = pd.read_feather(train_file)
        #df_train.set_index(df_train.columns[0])
        df_test = pd.read_feather(test_file)
        #df_test.set_index(df_test.columns[0])
        cutoff = int(df_train.shape[0] * (1-frac_val))
        df_val = df_train.iloc[cutoff:]
        df_train = df_train.iloc[:cutoff]
        print("df_train_shape", str(df_train.shape))
        print("df_val_shape", str(df_val.shape))
        print("df_test_shape", str(df_test.shape))


        if scaling:
            scaler = MinMaxScaler()
            scaler.fit(df_train.loc[:, original_features])
            df_train_scaled = pd.DataFrame(scaler.transform(df_train.loc[:, original_features]), columns=original_features, index=df_train.index)
            df_train = df_train.drop(original_features, axis=1)
            df_train = df_train.merge(df_train_scaled, left_index=True, right_index=True)
            df_val_scaled = pd.DataFrame(scaler.transform(df_val.loc[:, original_features]), columns=original_features, index=df_val.index)
            df_val = df_val.drop(original_features, axis=1)
            df_val = df_val.merge(df_val_scaled, left_index=True, right_index=True)
            df_test_scaled = pd.DataFrame(scaler.transform(df_test.loc[:, original_features]), columns=original_features, index=df_test.index)
            df_test = df_test.drop(original_features, axis=1)
            df_test = df_test.merge(df_test_scaled, left_index=True, right_index=True)

        print("df_train_shape", str(df_train.shape))
        print("df_val_shape", str(df_val.shape))
        print("df_test_shape", str(df_test.shape))
        df = pd.concat([df_train, df_val, df_test])
        print(df.shape)
        print(df.loc[:, column_dict['label']].value_counts())

        data = HeteroData()
        # determines the order of nodes
        tx_list = list(df.loc[:, column_dict['id']].unique())
        print(len(tx_list))
        tx_dict = {id:i for i, id in enumerate(tx_list)}

        card_list = list(df.loc[:, column_dict['sender']].unique())
        card_dict = {id:i for i, id in enumerate(card_list)}

        merchant_list = list(df.loc[:, column_dict['receiver']].unique())
        merchant_dict = {id:i for i, id in enumerate(merchant_list)}

        tx_dict = {id:i for i, id in enumerate(tx_list)}
        card_dict = {id:i for i, id in enumerate(card_list)}
        merchant_dict = {id:i for i, id in enumerate(merchant_list)}

        number_of_tx = len(list(df.loc[:, column_dict['id']].unique()))
        df_tx_index = df.set_index(column_dict['id'])

        
        #features = original_features.copy()
        #if rfm:
        #    features += rfm_features
        #if embeddings:
        #    features += embedding_features_tx
        #if apate:
        #    features += apate_features
        
        
        
        # Get y tensor
        # Get y tensor
        y = df_tx_index.loc[tx_list, column_dict['label']].values.astype(int)
        #b = np.zeros(len(card_list))
        #c = np.zeros(len(merchant_list))
        #y = np.concatenate((a,b,c))
        y = torch.tensor(y, dtype=torch.long)
        
        #print('done y')
        if weighted:
            edge_weight_col = df.loc[:, column_dict['edge_weight']]
            delta = min(edge_weight_col[edge_weight_col > 0]) / 100

        # The influence from cardholder/merchant on transaction nodes is not split according to label information!
        row = df.loc[:, column_dict['sender']].apply(lambda x: card_dict[x]).values
        col = df.TX_ID.apply(lambda x: tx_dict[x]).values
         #np.ones(len(df.loc[:, column_dict['sender']]))
        row = torch.from_numpy(row).to(torch.long)
        col = torch.from_numpy(col).to(torch.long)
        edge_index_CH = torch.stack([row, col], dim=0)
        if weighted:
            cell_data = df.loc[:, column_dict['edge_weight']].values
            cell_data = torch.from_numpy(cell_data.astype(np.double)).to(torch.double)
            edge_weight_CH = cell_data + delta
        else:
            edge_weight_CH = None
        
        row = df.loc[:, column_dict['receiver']].apply(lambda x: merchant_dict[x]).values
        col = df.TX_ID.apply(lambda x: tx_dict[x]).values
        row = torch.from_numpy(row).to(torch.long)
        col = torch.from_numpy(col).to(torch.long)
        edge_index_MC = torch.stack([row, col], dim=0)
        if weighted:
            cell_data = df.loc[:, column_dict['edge_weight']].values #np.ones(len(df.loc[:, column_dict['receiver']]))
            cell_data = torch.from_numpy(cell_data.astype(np.double)).to(torch.double)
            edge_weight_MC = cell_data + delta
        else:
            edge_weight_MC = None

            
        
        # get x
        tx_x = df_tx_index.loc[tx_list, original_features].values
        ch_x = np.zeros((len(card_list), 1))
        mc_x = np.zeros((len(merchant_list), 1))


        tx_x = torch.tensor(tx_x.astype(np.float), dtype=torch.float32)
        ch_x = torch.tensor(ch_x.astype(np.float), dtype=torch.float32)
        mc_x = torch.tensor(mc_x.astype(np.float), dtype=torch.float32)
        #print('done x')

        data['transaction'].x = tx_x 
        data['cardholder'].x = ch_x
        data['merchant'].x = mc_x

        #From cardholder and merchant to transaction nodes we do not distinguish between fraud/non-fraud.
        data['cardholder', 'pays', 'transaction'].edge_index = edge_index_CH
        data['merchant', 'receives', 'transaction'].edge_index = edge_index_MC

        data['cardholder', 'pays', 'transaction'].edge_weight = edge_weight_CH
        data['merchant', 'receives', 'transaction'].edge_weight = edge_weight_MC


    

        data['transaction'].y = y
                
        # Validation fraction from training data
        
        train_mask = torch.zeros(y.size(0), dtype=torch.bool)
        val_mask = torch.zeros(y.size(0), dtype=torch.bool)
        test_mask = torch.zeros(y.size(0), dtype=torch.bool)

        train_cutoff = int(df_train.shape[0])
        val_cutoff = int(df_train.shape[0]) + int(df_val.shape[0])

        train_mask[:train_cutoff] = True
        val_mask[train_cutoff:val_cutoff] = True
        test_mask[val_cutoff:number_of_tx] = True

        data['transaction'].train_mask = train_mask
        data['transaction'].val_mask = val_mask
        data['transaction'].test_mask = test_mask

        return data