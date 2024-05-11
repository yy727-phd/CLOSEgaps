from utils import *
import torch
import pandas as pd
import cobra
from CLOSEgaps import CLOSEgaps
import argparse

def train(feature, y, incidence_matrix, model, optimizer,loss_fun):
    model.train()
    optimizer.zero_grad()
    y_pred = model(feature, incidence_matrix)
    loss = loss_fun(y_pred, y)
    print(loss.item())
    loss.backward()
    optimizer.step()


def test(feature, incidence_matrix, model):
    model.eval()
    with torch.no_grad():
        y_pred = model.predict(feature, incidence_matrix)
    return torch.squeeze(y_pred)

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emb_dim', type=int, default=64)  # 32  #128
    parser.add_argument('--conv_dim', type=int, default=128) #128
    parser.add_argument('--L', type=int, default=2)  # 1
    parser.add_argument('--head', type=int, default=6)  # 2
    parser.add_argument('--p', type=float, default=0.1)  # 0.3
    parser.add_argument('--g_lambda', type=float, default=1)  # 0.3
    parser.add_argument('--num_iter', type=int, default=1)
    parser.add_argument('--max_epoch', type=int, default=90)  # 300
    parser.add_argument('--lr', type=float, default=1e-2)  # 0.01
    parser.add_argument('--weight_decay', type=float, default=1e-3)  # 5e-4
    return parser.parse_args()

def predict():
    print('-------------------------------------------------------')
    path = '../data/gems/draft_gems'
    namelist = get_filenames(path)
    args = parse()
    # read reaction pool
    bigg_pool = cobra.io.read_sbml_model('../data/pool/universe.xml')
    model_pool = cobra.io.read_sbml_model('./results/universe/comb_universe.xml')
    # change xml file to dataframe
    model_pool_df = create_stoichiometric_matrix(model_pool, array_type='DataFrame')
    for sample in namelist:
        if sample.endswith('.xml'):
            print('training HyperRXN and predicting reaction scores: ' + sample[:-4] + '...')
            # read the model and reaction pool
            rxn_df, rxn_pool_df = get_data_from_pool(path, sample, model_pool_df)
            incidence_matrix_pos = np.abs(rxn_df.to_numpy()) > 0
            incidence_matrix_pos = torch.tensor(incidence_matrix_pos, dtype=torch.float)
            incidence_matrix_pos = torch.unique(incidence_matrix_pos, dim=1)
            incidence_matrix_cand = np.abs(rxn_pool_df.to_numpy()) > 0
            incidence_matrix_cand = torch.tensor(incidence_matrix_cand, dtype=torch.float)
            score = torch.empty((incidence_matrix_cand.shape[1], args.num_iter))
            for i in range(args.num_iter):
                # create negative reactions
                incidence_matrix_neg = create_neg_incidence_matrix(incidence_matrix_pos)
                incidence_matrix_neg = torch.unique(incidence_matrix_neg, dim=1)
                incidence_matrix = torch.cat((incidence_matrix_pos, incidence_matrix_neg), dim=1)
                y = create_label(incidence_matrix_pos, incidence_matrix_neg)
                y = torch.tensor(y, dtype=torch.long)
                node_num, hyper_num = incidence_matrix.shape
                model = CLOSEgaps(input_num=node_num, input_feature_num=incidence_matrix_pos.shape[1], emb_dim=args.emb_dim, conv_dim=args.conv_dim,
                                    head=args.head, p=args.p, L=args.L,
                                    use_attention=True)

                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                crossentropyloss = torch.nn.CrossEntropyLoss()
                print(' --------- start training --------------------')
                for _ in range(args.max_epoch):
                    # trainings
                    model.train()
                    epoch_size = incidence_matrix.shape[1] // 10
                    l = 0
                    for itern in range(10):
                        optimizer.zero_grad()
                        y_pred = model(incidence_matrix_pos,
                                       incidence_matrix[:, itern * epoch_size:(itern + 1) * epoch_size])
                        loss = crossentropyloss(y_pred, y[itern * epoch_size:(itern + 1) * epoch_size])
                        loss.backward()
                        optimizer.step()
                        l += loss.item()
                    l = l / 10
                    print(f'epoch:{_}, loss:{l}')

                score[:, i] = test(incidence_matrix_pos, incidence_matrix_cand, model)[:, 1]
            score_df = pd.DataFrame(data=score.detach().numpy(), index=rxn_pool_df.columns)
            bigg_rxns = set([item.id for item in bigg_pool.reactions])
            common_rxns = list(bigg_rxns & set(score_df.index))
            common_score_df = score_df.T[common_rxns].T
            common_score_df.to_csv('./results/fba_result/' + sample[:-4] + '.csv')
    print('done for prediction!')

if __name__ == '__main__':
    predict()