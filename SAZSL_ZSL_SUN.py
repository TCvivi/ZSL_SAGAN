import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import numpy as np
import scipy.io
import torch.utils.data as Data
from sklearn.metrics import accuracy_score
import argparse
import os

class Dataset:
    def __init__(self, data_dir='../data/GBU', dataset='AWA1'):
        res101 = scipy.io.loadmat(data_dir + '/' + dataset + '/res101.mat')
        att_spilts = scipy.io.loadmat(data_dir + '/' + dataset + '/att_splits.mat')
        features = res101['features'].T
        labels = res101['labels'].astype(int).squeeze() - 1

        # spilt the features and labels
        trainval_loc = att_spilts['trainval_loc'].squeeze() - 1  # minus 1 for matlab is from 1 ranther than from 0
        test_seen_loc = att_spilts['test_seen_loc'].squeeze() - 1
        test_unseen_loc = att_spilts['test_unseen_loc'].squeeze() - 1
        
        # convert to torch tensor
        train = torch.from_numpy(features[trainval_loc]).float()
        train_label = torch.from_numpy(labels[trainval_loc]).unsqueeze(1)
        test_seen = torch.from_numpy(features[test_seen_loc]).float()
        test_seen_label = torch.from_numpy(labels[test_seen_loc]).unsqueeze(1)
        test_unseen = torch.from_numpy(features[test_unseen_loc]).float()
        test_unseen_label = torch.from_numpy(labels[test_unseen_loc]).unsqueeze(1)
        att = torch.from_numpy(att_spilts['att'].T).float()
        original_att = torch.from_numpy(att_spilts['original_att'].T).float()
        unseen_unique_label = torch.from_numpy(np.unique(test_unseen_label))
        
        # get the labels of seen classes and unseen classes
        seen_label = test_seen_label.unique()
        unseen_label = test_unseen_label.unique()
        
        # form the data as a dictionary
        self.data = {'train': train, 'train_label': train_label, #N
        'test_seen': test_seen, 'test_seen_label': test_seen_label, #M
        'test_unseen': test_unseen, 'test_unseen_label': test_unseen_label,
        'att': att,
        'unseen_unique_label':unseen_unique_label,
        'original_att': original_att,
        'seen_label': seen_label, # 40
        'unseen_label': unseen_label} # 10
        
        self.feature_size = train.shape[1]
        self.att_size = att.shape[1]
        self.seen_class = seen_label.shape[0]
        self.unseen_class = unseen_label.shape[0]
    
    def get_loader(self, opt='train', batch_size=32):
        data = Data.TensorDataset(self.data[opt], self.data[opt+'_label'])
        data_loader = Data.DataLoader(data, batch_size=batch_size, drop_last=True, shuffle=True)        
        return data_loader
    
class Generator(nn.Module):
    def __init__(self, feature_size=2048, att_size=85):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        
    def forward(self, att):# feature:, att:
        feature = torch.relu(self.fc1(att)) 
        feature = torch.sigmoid(self.fc2(feature))
        return feature
    
class Discriminator(nn.Module):  # the structure is artotally same with Generator
    def __init__(self, feature_size=2048, att_size=85):
        super(Discriminator, self).__init__()         
        self.fc1 = nn.Linear(att_size, 1024)
        self.fc2 = nn.Linear(1024, feature_size)
        self.fc1.bias.data.fill_(0)
        self.fc2.bias.data.fill_(0)
        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
    def forward(self, att):  # feature: , att:
        att_embed = torch.relu(self.fc1(att))
        att_embed = torch.relu(self.fc2(att_embed))
        return att_embed
    
def compute_D_acc(discriminator, dataset, batch_size=128, opt1='gzsl', opt2='test_seen'):
    test_loader = dataset.get_loader(opt2, batch_size=batch_size)
    att = dataset.data['att'].cuda()
    if opt1 == 'gzsl':
        search_space = torch.arange(att.shape[0])
    if opt1 == 'zsl':
        search_space = dataset.data['unseen_label']
    att = att[search_space].unsqueeze(0).repeat([batch_size, 1, 1])  # (B, test_class, 85)
    pred_label = []
    true_label = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.cuda(), labels.cuda()
#             features = F.normalize(features, p=2, dim=-1, eps=1e-12)
            features = features.unsqueeze(1).repeat(1, search_space.shape[0], 1)
            att_embed = discriminator(att).cuda() #bs 50 2048
            score = F.cosine_similarity(att_embed, features, dim=-1) 
            pred = torch.argmax(score, dim=1)
            pred = search_space[pred]
            pred_label = np.append(pred_label, pred.cpu().numpy())
            true_label = np.append(true_label, labels.cpu().numpy())

    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0

    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]
    return acc

def compute_G_acc(generator, dataset, opt1='gzsl', opt2='test_seen'):
    features = dataset.data[opt2].cuda() # 5685 * 2048
#     features = F.normalize(features, p=2, dim=-1, eps=1e-12)
    labels = dataset.data[opt2 + '_label'] # 5685 * 1
    att = dataset.data['att'].cuda() #  50 * 85
    if opt1 == 'gzsl':
        search_space = torch.arange(att.shape[0]) # 50
    if opt1 == 'zsl':
        search_space = dataset.data['unseen_label'] # 10
    pred_label = []
    true_label = []
    att = att[search_space] # (test_class_num, 85) 10*85 50*85   
    gen_fea = generator(att) # 10 * 2048
    features = F.normalize(features, p=2, dim=-1, eps=1e-12)
    gen_fea = F.normalize(gen_fea, p=2, dim=-1, eps=1e-12)
    result = features @ gen_fea.t() #5685 * 1 * 10
    pred = torch.argmax(result.squeeze(), 1) # 5685 * 1
    pred = search_space[pred]
    pred_label = np.append(pred_label, pred.cpu().numpy())
    true_label = np.append(true_label, labels.cpu().numpy())
    pred_label = np.array(pred_label, dtype='int')
    true_label = np.array(true_label, dtype='int')
    acc = 0
    unique_label = np.unique(true_label)
    for i in unique_label:
        idx = np.nonzero(true_label == i)[0]
        acc += accuracy_score(true_label[idx], pred_label[idx])
    acc = acc / unique_label.shape[0]
    return acc

def train(discriminator, generator, dataset, dlr=0.005, glr=0.005, batch_size=64, epochs=150, t=10.0, alpha=1.0, beta=1.0, save_path='model'):

    sigma = torch.tensor(t, requires_grad=True, device='cuda')
    feature_size, att_size, seen_class, unseen_class = dataset.feature_size, dataset.att_size, dataset.seen_class,dataset.unseen_class #2048,85,10
    att = dataset.data['att'].cuda() #50 * 85
    seen_att_original = att[dataset.data['seen_label']] # 40* 85
    seen_att = seen_att_original.unsqueeze(0).repeat([batch_size, 1, 1]) #for cls loss 32 * 40 * 85
    new_label = torch.zeros(att.shape[0]).long().cuda() # 50
    new_label[dataset.data['seen_label']] = torch.arange(seen_att.shape[1]).cuda() # new label for classification
    unseen_att = att[dataset.data['unseen_label']] #10 * 85

    #class average seen features
    train = dataset.data['train'].cuda() # 19832 * 2048
    train_label = dataset.data['train_label'] # 19832 * 1
    avg_feature = torch.zeros(att.shape[0], feature_size).float().cuda() # 50*2048
    cls_num = torch.zeros(att.shape[0]).float().cuda()
    seen_label = dataset.data['seen_label'].cuda() #40
    for i,l in enumerate(train_label):
        avg_feature[l] += train[i]
        cls_num[l] += 1
    for ul in seen_label:
        avg_feature[ul] = avg_feature[ul]/cls_num[ul]
        seen_avg_fea = avg_feature[seen_label] #40 * 2048

    train_loader = dataset.get_loader('train', batch_size)
    D_optimizer = optim.Adam(discriminator.parameters(), lr=dlr, weight_decay=0.00001)
    G_optimizer = optim.Adam(generator.parameters(), lr=glr, weight_decay=0.00001)
    D_scheduler = optim.lr_scheduler.StepLR(D_optimizer, 80, gamma=0.5)
    G_scheduler = optim.lr_scheduler.StepLR(G_optimizer, 80, gamma=0.5)
    entory_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()
    best_zsl = 0
    best_seen = 0
    best_unseen = 0
    best_H = 0
    best_zsl_G = 0
    best_seen_G = 0
    best_unseen_G = 0
    best_H_G = 0

    for epoch in range(epochs):
        print("Epoch {}/{}...".format(epoch + 1, epochs))
        D_loss = []
        G_loss = []

        for feature, label in train_loader:
            feature, label = feature.cuda(), label.cuda() 
            # train the discriminator
            D_optimizer.zero_grad()
            att_bs_seen = att[label] # bs * 85
            fea_fake_seen = generator(att_bs_seen) #bs * 2048
            att_embed = discriminator(att_bs_seen)
            pos_score = sigma * F.cosine_similarity(att_embed, feature, dim=-1)
            neg_score = sigma * F.cosine_similarity(att_embed, fea_fake_seen, dim=-1)
            pos_score = torch.mean(pos_score)
            neg_score = torch.mean(neg_score) 

            cls_features = feature.unsqueeze(1).repeat([1, seen_att.shape[1], 1]) 
            seen_att_D = discriminator(seen_att)
            cls_score = sigma * F.cosine_similarity(seen_att_D, cls_features, dim=-1)
            cls_label = new_label[label]
            cls_loss = entory_loss(cls_score, cls_label.squeeze())  

            d_loss =  - torch.log(pos_score) + torch.log(neg_score) + alpha * cls_loss  
            d_loss.backward(retain_graph=True)
            D_optimizer.step()
            D_scheduler.step(epoch)
            D_loss.append(d_loss.item())

            # train the generator
            G_optimizer.zero_grad()

            cls_fake_features = fea_fake_seen.repeat([1, seen_att.shape[1], 1]) # (32, 40, 2048)
            cls_fake_score = sigma * F.cosine_similarity(seen_att_D, cls_fake_features, dim=-1) 
            cls_fake_loss = entory_loss(cls_fake_score, cls_label.squeeze())     

            fea_fake_unseen = generator(unseen_att)  # 10 * 2048  
            att_matrix = att_bs_seen @ unseen_att.t() # 32*85 * 85*10 = 32 * 10
            att_m = F.softmax(att_matrix.squeeze(), dim=1)    
            fea_matrix_rf = feature @ fea_fake_unseen.t() #32*2048 * 2048*10 = 32 * 10
            fea_m_rf = F.softmax(fea_matrix_rf.squeeze(), dim=1)
            struct_loss_rf = torch.mean(mse_loss(att_m, fea_m_rf))

            fea_matrix_ff = fea_fake_seen @ fea_fake_unseen.t() #32*2048 * 2048*10 = 32 * 10
            fea_m_ff = F.softmax(fea_matrix_ff.squeeze(), dim=1)
            struct_loss_ff = torch.mean(mse_loss(att_m, fea_m_ff))

            #avg struct loss 
            all_att_matrix = seen_att_original @ unseen_att.t() # 40*85 * 85*10 = 40 * 10
            all_att_m = F.softmax(all_att_matrix.squeeze(), dim=1)    
            all_fea_matrix = seen_avg_fea @ fea_fake_unseen.t() #40*2048 * 2048*10 = 40 * 10
            all_fea_m = F.softmax(all_fea_matrix.squeeze(), dim=1)
            all_struct_loss = torch.mean(mse_loss(all_att_m, all_fea_m))

            g_loss = - torch.log(neg_score) + alpha * cls_fake_loss + beta * (struct_loss_rf + struct_loss_ff + all_struct_loss)
            g_loss.backward()
            G_optimizer.step()
            G_scheduler.step(epoch)
            G_loss.append(g_loss.item())

            D_scheduler.step()
            G_scheduler.step()

            # test
            D_zsl_acc = compute_D_acc(discriminator, dataset, batch_size = batch_size, opt1='zsl', opt2='test_unseen')
            D_seen_acc = compute_D_acc(discriminator, dataset, batch_size = batch_size, opt1='gzsl', opt2='test_seen')
            D_unseen_acc = compute_D_acc(discriminator, dataset, batch_size = batch_size, opt1='gzsl', opt2='test_unseen')
            D_harmonic_mean = (2 * D_seen_acc * D_unseen_acc) / (D_seen_acc + D_unseen_acc)
            best_zsl = D_zsl_acc if D_zsl_acc > best_zsl else best_zsl
            if D_harmonic_mean > best_H:
                best_H = D_harmonic_mean
                best_seen = D_seen_acc
                best_unseen = D_unseen_acc
            
            G_zsl_acc = compute_G_acc(generator, dataset, opt1='zsl', opt2='test_unseen')
            G_seen_acc = compute_G_acc(generator, dataset, opt1='gzsl', opt2='test_seen')
            G_unseen_acc = compute_G_acc(generator, dataset, opt1='gzsl', opt2='test_unseen')
            G_harmonic_mean = (2 * G_seen_acc * G_unseen_acc) / (G_seen_acc + G_unseen_acc)

            best_zsl_G = G_zsl_acc if G_zsl_acc > best_zsl_G else best_zsl_G
            if G_harmonic_mean > best_H_G:
                best_H_G = G_harmonic_mean
                best_seen_G = G_seen_acc
                best_unseen_G = G_unseen_acc

        print('D_loss:{:.3f},unseen:{:.3f},seen:{:.3f},Best_H:{:.3f}'.format(np.mean(D_loss),best_unseen, best_seen, best_H))
        print('G_loss:{:.3f},unseen:{:.3f},seen:{:.3f},Best_H:{:.3f}'.format(np.mean(G_loss),best_unseen_G, best_seen_G, best_H_G))
        torch.save(discriminator,os.path.join(save_path, "D_"+str(epoch+1)+"_weight.pth"))
        torch.save(generator,os.path.join(save_path, "G_"+str(epoch+1)+"_weight.pth"))
    
def main(opt):
    dataset = Dataset(data_dir=opt.data_dir, dataset=opt.dataset)
    discriminator = Discriminator(dataset.feature_size, dataset.att_size).cuda()
    generator = Generator(dataset.feature_size, dataset.att_size).cuda()
    train(discriminator, generator, dataset, dlr=opt.d_lr, glr=opt.g_lr, batch_size=opt.batch_size, epochs=opt.epochs, t=opt.t, alpha=opt.alpha, beta=opt.beta, save_path=opt.save_path) 

if __name__ == '__main__':
    if not torch.cuda.is_available():
        print('please use GPU!')
        exit()
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='SUN')
    parser.add_argument('--data_dir', default='../data/GBU')
    parser.add_argument('--d_lr', type=float, default=0.005)
    parser.add_argument('--g_lr', type=float, default=0.005)
    parser.add_argument('--t', type=float, default=20.0)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--beta', type=float, default=10.0)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--save_path', default='model20200927')

    opt = parser.parse_args()
    main(opt)
