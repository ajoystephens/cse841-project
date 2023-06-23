import numpy as np
import pandas as pd
import pickle
import torch 
import torch.nn.functional as F
from torch.nn import Linear
from sklearn.metrics import accuracy_score,f1_score, roc_auc_score
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix
from sklearn.preprocessing import normalize

from aif360.sklearn.metrics import statistical_parity_difference, average_odds_difference

from argparse import ArgumentParser


# ==============================================================================
#  SETUP
# ==============================================================================
argParser = ArgumentParser()
argParser.add_argument("-d", "--dataset", default='compas', help="dataset")
argParser.add_argument("-sub", "--subset", default='clean', help="could be clean, dirty, or corrected")
argParser.add_argument("-e", "--epochs", default=50, type=int, help="number of epochs")
argParser.add_argument("-lr", "--learning_rate", default=0.05, help="learning_rate")
argParser.add_argument("-hi", "--hidden", default=32, help="dimmensions of hidden layer")
argParser.add_argument("-dr", "--dropout", default=0.2, help="dropout")
argParser.add_argument("-pt", "--p_test", default=0.2, help="percent of dataset to set aside for testing")
argParser.add_argument("-pv", "--p_val", default=0.2, help="percent of dataset to set aside for validation")

args = argParser.parse_args()

print(f'====={" SETUP " :=<85}')
print(f'-----{" Initial Args " :-<85}')
for arg in vars(args):
    print(f'\t{arg}: {getattr(args, arg)}')

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
SEEDS = [1,5,10,15,20,25,30,35,40,45]


print(f'-----{" Other " :-<85}')
print(f'DEVICE: {DEVICE}')


class Simple(torch.nn.Module):
    def __init__(self, dim_in, hidden,dim_out,lr,dropout):
        super().__init__()
        self.l1 = Linear(dim_in, hidden)
        self.l2 = Linear(hidden,dim_out)
        self.optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        self.dropout = dropout


    def forward(self, x):
        h = self.l1(x)
        h = torch.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = self.l2(h)
        h = torch.relu(h)
        return h, F.softmax(h,dim=1)

def train(model, x, y, train_mask, val_mask, epochs):
    """Train a GNN model and return the trained model."""
    train_losses = []
    val_losses = []
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = model.optimizer

    model.train()
    for epoch in range(epochs+1):

        # Training
        optimizer.zero_grad()

        out, out_softmax = model(x[train_mask])
        
        print(train_mask.shape)
        print(y.shape)
        print(out.shape)
        loss = criterion(out, y[train_mask])
        quit()
        out_softmax = out_softmax.clone().detach().cpu().numpy()
        pred = out_softmax.argmax(axis=1)

        # pred_np = pred.clone().detach().cpu().numpy()
        y_np = y.clone().detach().cpu().numpy().argmax(axis=1)


        loss.backward()
        optimizer.step()

        with torch.no_grad():
            model.eval()
            out, out_softmax = model(x[val_mask])
            val_loss = criterion(out, y[val_mask])
            val_softmax = out_softmax.clone().detach().cpu().numpy()
            val_pred = val_softmax.argmax(axis=1)
            val_cm = confusion_matrix(y_np[val_mask],val_pred)

            # val_pred_np = val_pred.clone().detach().cpu().numpy()


        # Print metrics every 10 epochs
        if(epoch % 10 == 0):
            print(f'Epoch {epoch:>3} | Train Loss: {loss:.3f} | Val Loss: {val_loss:.3f} | ')
            print(f'val confusion matrix:\n{val_cm}')
            print(val_softmax[1,:])
            

    return model

def test(model, x, y, test_mask):
    """Evaluate the model on test set and print the accuracy score."""


    model.eval()
    with torch.no_grad():

        out, out_softmax = model(x[test_mask])
        out_softmax = out_softmax.clone().detach().cpu().numpy()
        pred = out_softmax.argmax(axis=1)
        # pred_np = pred.clone().detach().cpu().numpy()
        y_np = y.clone().detach().cpu().numpy().argmax(axis=1)

        acc = accuracy_score(y_np[test_mask],pred)

    return pred,y_np[test_mask]


# ==============================================================================
#  GET DATA
# ==============================================================================
# temp = f" GETTING DATA:"
print(f'====={" GETTING DATA:" :=<85}')

# option = 'corrected' if args.corrected else 'clean'
x_path = f'datasets/{args.dataset}/{args.subset}.csv'
print(f'retrieving: {x_path}')

x = pd.read_csv(x_path, sep=",", header=0, dtype=float)
x = x.to_numpy()
x = np.nan_to_num(x)
x = normalize(x,axis=0)
x = torch.tensor(x,dtype=torch.float).to(DEVICE)

p_path = f'datasets/{args.dataset}/data.p'
print(f'retrieving: {p_path}')
file = open(p_path, 'rb')
p_data = pickle.load(file)
file.close()
y = torch.tensor(p_data['Y'],dtype=torch.float).to(DEVICE)
p = torch.tensor(p_data['P'],dtype=torch.float).to(DEVICE)
priv_group = p_data['priv_group']
pos_label = p_data['pos_label']

n = x.shape[0]

print(f'X shape: {x.shape}')
print(f'Y shape: {y.shape}')
print(f'P shape: {p.shape}')



# ==============================================================================
#  TRAIN
# ==============================================================================
results={
    'acc':[],
    'sp':[],
    'eo':[],
}
for seed in SEEDS:
    temp = f" BEGIN SEED: {seed} "
    print(f'====={temp :=<85}')

    # set seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # split the data
    n_test = int(n * args.p_test)
    n_val = int(n * args.p_val)
    n_train = int(n - n_test - n_val)
    ind = np.arange(n)
    np.random.shuffle(ind)
    i_test = ind[:n_test]
    i_val = ind[n_test:(n_test+n_val)]
    i_train = ind[(n_test+n_val):]

    train_mask = np.zeros(n,dtype=bool)
    train_mask[i_train] = True
    val_mask = np.zeros(n,dtype=bool)
    val_mask[i_val] = True
    test_mask = np.zeros(n,dtype=bool)
    test_mask[i_test] = True

    print(f'Train Cnt: {n_train}')
    print(f'Val Cnt: {n_val}')
    print(f'Test Cnt: {n_test}')


    # Train
    print(f'-----{" Training Model " :-<85}')
    model = Simple(x.shape[1], args.hidden,y.shape[1],args.learning_rate,args.dropout).to(DEVICE)
    train(model, x, y, train_mask, val_mask, args.epochs)

    print(f'-----{" Testing Model " :-<85}')
    # Test
    pred,label = test(model, x, y, test_mask)
    prot = p[test_mask].clone().detach().cpu().numpy()


    acc = accuracy_score(label,pred)
    sp = statistical_parity_difference(label,pred,prot_attr=prot,pos_label=0)
    eo = average_odds_difference(label,pred,prot_attr=prot,priv_group=priv_group,pos_label=pos_label)
    cm = confusion_matrix(label,pred)
    print(f'test confusion matrix:\n{cm}')
    print(f'accuracy: {acc}')
    print(f'stat parity: {sp}')
    print(f'equal odds: {eo}')

    results['acc'].append(acc)
    results['sp'].append(sp)
    results['eo'].append(eo)



    print(f'-----{"-" :*<85}')


print(f'====={" CHECK ALL RESULTS:" :=<85}')
for key in results:
    mean = np.average(results[key])
    std = np.std(results[key])
    print(f'{key}: {mean:.5f} +/- {std:.5f}')
