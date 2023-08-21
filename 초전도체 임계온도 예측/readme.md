# 데이콘 초전도체 임계온도 예측 
* 링크: [초전도체 임계온도 예측](https://dacon.io/competitions/official/236146/overview/description)
* [private 1위](https://dacon.io/competitions/official/236146/codeshare/8700?page=1&dtype=recent)
* [private 2위](https://dacon.io/competitions/official/236146/codeshare/8699?page=1&dtype=recent)
* [private 4위](https://dacon.io/competitions/official/236146/codeshare/8697?page=1&dtype=recent)
* [private 5위](https://dacon.io/competitions/official/236146/codeshare/8698?page=1&dtype=recent)
* [private 6위](https://dacon.io/competitions/official/236146/codeshare/8695?page=1&dtype=recent)



# 내 방법론
``` python
######### 데이터 sampling

from torch.utils.data import DataLoader

class CustomDataset(Dataset):
  def __init__(self, train_df, *args, **kwards):
    self.feature = train_df.drop('critical_temp', axis = 1).values
    self.label = train_df['critical_temp'].values.reshape(-1, 1) # 둘의 차원(feature와 label)이 같아야 한다.
    #print(self.label)
  def __len__(self):
    return len(self.label)

  def __getitem__(self, index):
    x = torch.FloatTensor(self.feature[index])
    y = torch.FloatTensor(self.label[index])
    return x, y


def partition(train_df, val_size):
  num_total = train_df.shape[0]
  num_train = int((1 - val_size)*num_total)
  num_val = int(num_total * val_size)
  df = train_df.sample(frac=1).reset_index(drop=True)  # shuffling하고 index reset

  train_split = df[:num_train]
  valid_split = df[num_train:]

  train_set = CustomDataset(train_split, args)
  valid_set = CustomDataset(valid_split, args)

  partition = {'train': train_set, 'valid': valid_set}

  return partition
```

```python
class GatedSkipConnection(nn.Module):
  def __init__(self, in_dim, out_dim):
    super(GatedSkipConnection, self).__init__()
    self.linear = nn.Linear(in_dim, out_dim, bias=False)
    self.linear_coef_in = nn.Linear(out_dim, out_dim)
    self.linear_coef_out = nn.Linear(out_dim, out_dim)
    self.sigmoid = nn.Sigmoid()
    self.in_dim = in_dim
    self.out_dim = out_dim

  def forward(self, in_x, out_x):
    #print('test gate')
    if (self.in_dim != self.out_dim):
      in_x = self.linear(in_x)
    z = self.gate_coefficient(in_x, out_x)
    out = torch.mul(z, out_x) + torch.mul(1.0-z, in_x)
    return out

  def gate_coefficient(self, in_x, out_x):
    x1 = self.linear_coef_in(in_x)
    x2 = self.linear_coef_out(out_x)
    return self.sigmoid(x1+x2)
```


```python
#from torch.autograd import Variable
class NNLayer(nn.Module):
  def __init__(self, in_dim, out_dim, act=None, bn=False, atn=False, num_head=1, dropout=0):
    super(NNLayer, self).__init__()

    # self.in_dim = in_dim
    # self.out_dim = out_dim
    # self.activation = nn.ReLU()
    # self.lstm = nn.LSTM(in_dim, out_dim)
    #___________________________________________
    self.use_bn = bn
    self.use_bn = bn
    self.use_atn = atn
    self.linear = nn.Linear(in_dim, out_dim)

    nn.init.xavier_uniform_(self.linear.weight)
    self.bn = nn.BatchNorm1d(out_dim)
    self.attention = Attention(out_dim, out_dim, num_head)

    self.activation = act
    self.dropout_rate = dropout
    self.dropout = nn.Dropout(self.dropout_rate)

  def forward(self, x):

    ##-------------------
    # h_0 = Variable(torch.zeros(x.shape[0], self.out_dim)).to(device)
    # c_0 = Variable(torch.zeros(x.shape[0], self.out_dim)).to(device)

    # print('check')
    # output, (hn, cn) = self.lstm(x, (h_0, c_0))

    # print(hn.shape)

    # out = hn.view(-1, self.out_dim)
    #--------------------------------
    #print(x)
    out = self.linear(x)
    #print(out.shape)
    if self.use_atn:
      out = self.attention(out)
   # print(out.shape)
    if self.use_bn:
      out = self.bn(out)
    if self.activation != None:
      out = self.activation(out)
    if self.dropout_rate >0:
      out = self.dropout(out)
    #print('nnlayer 끝')
    return out
```

```python
class NNBlock(nn.Module):
  def __init__(self, n_layer, in_dim, hidden_dim, out_dim, bn=True, atn=True, num_head=1, sc='gsc', dropout=0):
    super(NNBlock, self).__init__()

    self.layers=nn.ModuleList()

    for i in range(n_layer):
      self.layers.append(NNLayer(in_dim if i==0 else hidden_dim,
                                 out_dim if i==n_layer-1 else hidden_dim,
                                 nn.ReLU() if i!= n_layer-1 else None,
                                 bn,
                                 atn,
                                 num_head,
                                 dropout))
    self.relu=nn.ReLU()
    if sc=='gsc':
      self.sc = GatedSkipConnection(in_dim, out_dim)
    elif sc=='sc':
      self.sc = SkipConnection(in_dim, out_dim)
    elif sc=='no':
      self.sc = None
    else:
      assert False, "Wrong sc type."

  def forward(self, x):
    residual = x
    for i, layer in enumerate(self.layers):
      out = layer(x if i==0 else out)
    if self.sc != None:
      out = self.sc(residual, out)
    out = self.relu(out)
    #print('NNBLock')
    return out

```

```python
class ReadOut(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(ReadOut, self).__init__()

        self.in_dim = in_dim
        self.out_dim= out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
        #print(x.shape)
        out = self.linear(x)
        #print(out.shape)
        #out = torch.sum(out, )
        if self.activation != None:
            out = self.activation(out)
        return out
```
```python
class Predictor(nn.Module):

    def __init__(self, in_dim, out_dim, act=None):
        super(Predictor, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.linear = nn.Linear(self.in_dim,
                                self.out_dim)
        nn.init.xavier_uniform_(self.linear.weight)
        self.activation = act

    def forward(self, x):
      #print('predictor shape', x.shape)
      out = self.linear(x)
      if self.activation != None:
        out = self.activation(out)
      return out
```
```python
class NNNet(nn.Module):

    def __init__(self, args):
        super(NNNet, self).__init__()


        if args.pre_process == True:
          self.prepro = nn.ModuleList()
          for i in range(args.n_pre):
            self.prepro.append(ReadOut(args.in_dim if i ==0 else args.hidden_dim , args.hidden_dim if i == args.n_pre-1 else args.hidden_dim, act = nn.Tanh() if i == args.n_pre-1 else nn.ReLU()))

        self.blocks = nn.ModuleList()
        for i in range(args.n_block):
            self.blocks.append(NNBlock(args.n_layer,
                                        args.in_dim if i==0 else args.hidden_dim,# <- in_dim에서 hidden_dim 으로 수정
                                        args.hidden_dim,
                                        args.hidden_dim,
                                        args.bn,
                                        args.atn,
                                        args.num_head,
                                        args.sc,
                                        args.dropout))
        self.readout = ReadOut(args.hidden_dim,
                               args.pred_dim1,
                               act=nn.ReLU())
        self.pred1 = Predictor(args.pred_dim1,
                               args.pred_dim2,
                               act=nn.ReLU())
        self.pred2 = Predictor(args.pred_dim2,
                               args.pred_dim3,
                               act=nn.Tanh())
        self.pred3 = Predictor(args.pred_dim3,
                               args.out_dim)

    def forward(self, x):
        if args.pre_process==True:
          for i, block in enumerate(self.prepro):
            x = block(x)

        for i, block in enumerate(self.blocks):
            out = block(x if i==0 else out)
           #print('Net ', i)
            #print(out.shape)

        out = self.readout(out)
        #print('readout')
        out = self.pred1(out)
        #print('pred1')
        out = self.pred2(out)
        #print('pred2')
        out = self.pred3(out)
        #print('pred3')
        return out
```
```python
#파라미터
args.n_pre=2 # 추가
args.pre_process=False# 추가

args.batch_size = 100
args.lr = 0.001
args.l2_coef = 0
args.optim = 'Adam'
args.epoch = 200
args.n_block =2
args.n_layer = 1
#args.n_atom = 50
args.in_dim = train_df_final.shape[1]-1
args.hidden_dim = 512
args.pred_dim1 = 256
args.pred_dim2 = 128
args.pred_dim3 = 128
args.out_dim = 1
args.bn = True
args.sc = 'gsc'
args.atn = False
args.num_head = 1
args.step_size = 4
args.gamma = 0.95 # 0.1 -> 0.2, 0.3: best 0.6: 6.22 0.7: 6.0 정도 0.8: 5.9, 0.9: 5.6 0.95: 5.6, 0.92 5.4
args.dropout = 0
args.val_size = 0.1
args.shuffle=True
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
```

```python
# seed
import torch
import random
import torch.backends.cudnn as cudnn

torch.manual_seed(4)
torch.cuda.manual_seed(4)
torch.cuda.manual_seed_all(4)
np.random.seed(4)
cudnn.benchmark = False
cudnn.deterministic = True # 후보 1: 5.5 # 후보 seed 4: 5.0(gamma 95), 최고 4
random.seed(4)
```
