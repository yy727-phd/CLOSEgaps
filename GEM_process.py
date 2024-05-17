import argparse
import os
import pandas as pd
import cobra

from process_data import change_metaid_to_metaname, change_arrow
import numpy as np
from rdkit import Chem
from rdkit.Chem.Lipinski import HeavyAtomCount

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--GEM_name", type=str, default="iMM904")
    return parser.parse_args()

def get_filenames(path):
    return sorted(os.listdir(path))


def remove_rxn(model, name_list):
    remove_list = []
    for i in range(len(model.metabolites)):
        meta = model.metabolites[i]
        if meta.name in name_list:
            continue
        remove_list.append(meta)
    model.remove_metabolites(remove_list, destructive=True)
    print(f'remove_rxn:{len(remove_list)}')


def get_data(path, sample):
    model = cobra.io.read_sbml_model(path + '/' + sample)
    return model


def remove_right_empty(path='../iMM904', output_rxn_file=None, output_meta_file=None):
    print('-------------------------------------------------------')
    print('reading GEMs  ...')
    namelist = get_filenames(path)
    rxns_list = []
    for sample in namelist:
        if sample.endswith('xml'):
            model = get_data(path, sample)
            rxn_equation = np.array([rxn for rxn in model.reactions])
            for rxn in rxn_equation:
                id, rxn = str(rxn).split(': ')
                if '<--' in rxn:
                    left, right = rxn.split('<--')
                    rxn = right + ' => ' + left
                tem_rxn = rxn.replace('-->', '=>')
                dir = '=>'
                if '<=>' in tem_rxn:
                    dir = '<=>'
                print(tem_rxn)
                left, right = tem_rxn.split(dir)
                if right == ' ':
                    continue
                tem_rxn = tem_rxn.strip()
                rxns_list.append(tem_rxn)
            rxn_equation_list_df = pd.DataFrame({'rxn_equation': rxns_list})

            metas_id = np.array([meta.id for meta in model.metabolites])
            metas_id_df = pd.DataFrame({'name_id': metas_id})
            metas = np.array([meta.name for meta in model.metabolites])
            metas_name_df = pd.DataFrame(data=metas[0:], columns=['name'])
            metas_links = np.array([meta.annotation for meta in model.metabolites])
            metas_links_df = pd.DataFrame({'links': metas_links})
            metas = pd.concat([metas_id_df, metas_name_df, metas_links_df], axis=1)

            if output_rxn_file is not None:
                rxn_equation_list_df.to_csv(output_rxn_file, index=False)
            if output_meta_file is not None:
                metas.to_csv(output_meta_file, index=False)
    return rxns_list, metas


def get_chebi_link(metas, output_meta_chebi_file=None):
    valid_metas = metas[metas['links'].apply(lambda x: 'chebi' in x)]
    valid_metas.reset_index()
    valid_metas_re = valid_metas.reset_index(drop=True)
    l = []
    for i in range(valid_metas_re.shape[0]):
        p = valid_metas_re.links[i]['chebi']
        if not isinstance(p, list):
            l.append([p])
        else:
            l.append(p)
    valid_metas_re['chebi'] = l
    if output_meta_chebi_file is not None:
        valid_metas_re.to_csv(output_meta_chebi_file, index=False)
    return valid_metas_re


def get_smiles(all_metas, output_smiles_file=None):
    data = pd.read_csv('./data/pool/cleaned_chebi.csv')
    smiles = {'name': [], 'smiles': [], 'name_id': []}
    for i, row in all_metas.iterrows():
        name = row['name']
        name_id = row['name_id']
        s = ''
        for che in row['chebi']:
            try:
                s = data[int(che.split(':')[-1]) == data['ChEBI_ID']].smiles.values[0]
                break
            except Exception:
                pass
        smiles['name'].append(name)
        smiles['smiles'].append(s)
        smiles['name_id'].append(name_id)
    meta_smiles = pd.DataFrame(smiles)
    meta_smiles.drop_duplicates(inplace=True, ignore_index=True)
    meta_smiles = meta_smiles[meta_smiles['smiles'] != '']
    if output_smiles_file is not None:
        meta_smiles.to_csv(output_smiles_file, index=False)
    return meta_smiles


def cout_atom_number(metas, output_meta_count_file=None):
    results = pd.DataFrame([], columns=['name', 'smiles', 'count'])
    smiles = []
    count = []
    # metas2 = pd.read_csv('process/iMM904_meta_id_smiles_list.csv')
    metas['mol'] = metas['smiles'].apply(lambda x: Chem.MolFromSmiles(x))
    metas = metas[metas['mol'].isna() == False]
    metas.loc[:, 'smiles'] = metas['mol'].apply(lambda x: Chem.MolToSmiles(x))
    metas.loc[:, 'count'] = metas['mol'].apply(lambda x: HeavyAtomCount(x))
    metas = metas.drop(columns=['mol'])
    metas_remove_dup = metas.drop_duplicates('name', ignore_index=True)
    if output_meta_count_file is not None:
        metas_remove_dup.to_csv(output_meta_count_file, index=False)
    # metas_remove_dup.to_csv('./iMM904_meta_count.csv', index=False)
    return metas_remove_dup


if __name__ == '__main__':
    args = parse()
    gem_name = args.GEM_name
    print(gem_name)
    rxn_list_no_empty, all_metas = remove_right_empty(path=f'./data/{gem_name}',
                                                      output_rxn_file=f'./data/{gem_name}/{gem_name}_rxn_no_empty.csv')
    all_metas_name = all_metas.loc[:, ['name']]
    all_metas_remove_dup = all_metas_name.drop_duplicates()

    ## some 6e-06 in the rxn
    meta_chebi_link = get_chebi_link(all_metas)
    meta_smiles = get_smiles(meta_chebi_link)
    meta_smiles_count = cout_atom_number(meta_smiles,
                                         output_meta_count_file=f'./data/{gem_name}/{gem_name}_meta_count.csv')

    rxn_list_no_empty_clean = pd.read_csv(f'./data/{gem_name}/{gem_name}_rxn_no_empty.csv')['rxn_equation'].tolist()
    ## change rxn_id to rxn_name, and remove dup ##
    pos_rxns_names = change_metaid_to_metaname(rxn_list_no_empty_clean, all_metas)
    rxn_all_name_sample = pd.DataFrame({'rxn_names': pos_rxns_names})
    rxn_all_name_sample_clean = rxn_all_name_sample.drop_duplicates()

    ## change rxn_arrow, and get the rxn has smiles ##
    pos_rxns = change_arrow(pos_rxns_names, filter_name=meta_smiles_count['name'].tolist(),
                            save_file=f'./data/{gem_name}/{gem_name}_rxn_name_list.txt')

