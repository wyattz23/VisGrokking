import lightning as pl
#from datasets import load_dataset
import torch
from torch.utils.data import Dataset
import functools
import random
vocab= {"pad":0, "a":1, "b":2, "eos": 3}
#only allowing actions forward (b) (+1), or stay (a) (+0), cycle is closed at $n$-th position

#for example: 2 states (0, 1)
#starting position is state 0
#reading a : 0
#reading b: 1    
#reading b: 0
# equivalent b % 2 


#5 states (0,1,2,3,4)
#starting position is state 0
#reading a : 0
#reading b: 1
#reading b: 2
#reading b: 3
#reading b: 4
#reading b: 0
# equivalent b % 5 
class ParityTrain(Dataset):
    def __init__(self, max_len, m):
        #m: number of nodes in state machine
        #m = 2: parity 
        self.max_len = max_len
        self.m = m
    
        
    def __len__(self):
        return 10000
    def __getitem__(self, idx):
        l = random.randint(1, self.max_len) #length sampling
        input = ''.join(random.choice(['a', 'b']) for _ in range(l)) 
        count = input.count("b")
        # if count % 2 == 0:
        #     label = 1 #paired
        # else:
        #     label = 0 

        label = count % self.m
        sample = {"input": input, "label": label, "mask" : [1 for _ in range(l)]}
        return sample
    
class ParityTest(Dataset):
    def __init__(self, max_len, m):
        self.max_len = max_len
        self.m = m
    def __len__(self):
        return 1000
    def __getitem__(self, idx):
        l = random.randint( self.max_len+ 1 ,self.max_len* 4 )
        input = ''.join(random.choice(['a', 'b']) for _ in range(l))
        count = input.count("b")
        # if count % 2 == 0:
        #     label = 1 #paired
        # else:
        #     label = 0 
        label = count % self.m
        sample = {"input": input, "label": label}
        return sample

class MyDataLoader(pl.LightningDataModule):
    def __init__(
        self,
        max_len,
        n_workers,
        train_batch_size,
        test_batch_size,
        m
    ):
        super().__init__()
        self.max_len = max_len
        self.train_dataset = ParityTrain(max_len,m)
        self.val_dataset = ParityTest(max_len,m)
        #self.tokenizer =   AutoTokenizer.from_pretrained("nferruz/ProtGPT2")
        self.n_workers = n_workers 
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.collat = functools.partial(
                prepare_batch,
                max_len = self.max_len
                
            )
        
    
    def train_dataloader (self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size = self.train_batch_size, collate_fn = self.collat, pin_memory=True,
            num_workers=self.n_workers, shuffle=True )
    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size = self.test_batch_size, collate_fn = self.collat, pin_memory=True,
            num_workers=self.n_workers,)
def prepare_batch(batch, max_len, eos_idx = 0 ):
    
    input_ids_list = []
    attention_list = []
    target_list = []
    for each in batch:
        text = each["input"]
        input_ids = [vocab[each] for each in text] + [vocab["eos"]]
        attention_mask = [1 for _ in range(len(input_ids))]
        
        
        
        
        input_ids_list.append(torch.tensor(input_ids))
        target_list.append(each["label"])
        attention_list.append(torch.tensor(attention_mask))
        
        
        
    input_tensor = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first= True, padding_value=0)
    target_tensor = torch.tensor(target_list)
    mask_tensor = torch.nn.utils.rnn.pad_sequence(attention_list, batch_first= True, padding_value=0)
    return input_tensor, target_tensor, mask_tensor
        
        
# loader = ProteinDataLoader(1024, 8, 128 )
# for each in loader.train_dataloader:
#     print(each)
#     break