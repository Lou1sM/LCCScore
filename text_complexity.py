import os
import nltk
from nltk.corpus import treebank
import re
import json
import sys; sys.setrecursionlimit(110000)
import numpy as np
import pandas as pd
import argparse
from LCC.discrete_lcc import discrete_lcc


if __name__ == '__main__':
    all_dsets = ['text-en', 'text-de', 'text-ie', 'wsj', 'simp-en', 'rand', 'repeat2', 'repeat5', 'repeat10']
    all_dsets += [f'childes-{age}-{lang}' for age in (3,5,7) for lang in ['en']]
    all_wiki_topics = [
            'animals',
            'aurora',
            'chemistry',
            'computation',
            'life',
            'music',
            'ocean',
            'plants',
            'trees',
            'water',
            ]

    parser = argparse.ArgumentParser()
    parser.add_argument('--dsets', '-d', type=str, nargs='+', choices=all_dsets+['all'], required=True)
    parser.add_argument('--nits', type=int, default=1)
    parser.add_argument('--text-len', type=int, default=10000)
    parser.add_argument('--n-texts', type=int, default=1)
    parser.add_argument('--offset', type=int, default=0)
    parser.add_argument('--verbose', action='store_true')
    ARGS = parser.parse_args()

    dsets = all_dsets if 'all' in ARGS.dsets else ARGS.dsets

    def rand_tiled(n, N):
        totile = ''.join(np.random.choice(list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '), size=n))
        return ''.join(totile for _ in range(N//n))

    def get_wiki_text(topic, lang):
        with open(f'data/text/{topic}-{lang}.txt') as f:
            dset_text_ = f.read()
        dset_text = re.sub(r'\s+', ' ', dset_text_)
        dset_text = dset_text[:ARGS.text_len]
        if len(dset_text) < ARGS.text_len:
            print(f'text only length {len(dset_text)} for {topic}-{lang}')
        return dset_text

    def get_wsj(desired_len):
        nltk.download('treebank')
        corpus = ''
        fields = iter(treebank.fileids())
        while True:
            fld = next(fields)
            corpus += ' '.join(treebank.words(fld))
            if len(corpus) > desired_len:
                break
        return corpus

    def get_childes_by_age(age, lang):
        children = []
        age_dir = f'data/childes-{lang}/{age}'
        for fname in os.listdir(age_dir):
            with open(os.path.join(age_dir, fname)) as f:
                child = f.read().split('\n')
            child = '. '.join(line.removeprefix('*CHI:\t').removesuffix(' .') for line in child if line.startswith('*CHI:\t'))
            children.append(child)

        corpus = '\n'.join(children)
        corpus = re.sub(r'\[(\?|/|//|\*)\]', '', corpus)
        #corpus = re.sub(r'\[=\? ([A-Za-z\' ]+)\]', r'\1', corpus)
        corpus = re.sub(r'\([A-Za-z ]+\)', '',  corpus)
        corpus = re.sub(r'\.\.+ ', '.', corpus)
        corpus = re.sub(r'\. (\. )+', ' ', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*&[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*\+[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*<[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*>[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*:[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'[A-Za-z-~=:/\.\*]*@[A-Za-z-~=:/\.\*]*', '', corpus)
        corpus = re.sub(r'\(([A-Za-z\.\*]*)\)', r'\1', corpus)
        corpus = re.sub(r'[oauh]*\. ', ' ', corpus)
        corpus = re.sub(r'\[[A-Za-z-~=:/?\*\' ]+\]', '', corpus)
        corpus = re.sub(r'\([A-Za-z-~=:/?\*\' ]+\)', '', corpus)
        corpus = corpus.replace('O. ', '')
        corpus = corpus.replace('O ', '')
        corpus = corpus.replace(' \' ', '')
        corpus = corpus.replace('[?] ', '')
        corpus = corpus.replace('?. ', '? ')
        corpus = corpus.replace(' ?', '? ')
        corpus = re.sub(r' *, ', ', ', corpus)
        corpus = corpus.replace('„ ', ' ')
        corpus = corpus.replace('“', '')
        corpus = corpus.replace('”', '')
        corpus = corpus.replace('0 ', ' ')
        corpus = corpus.replace(' 0', ' ')
        corpus = re.sub(r' +', ' ', corpus)
        corpus = re.sub(r' +\n', '\n', corpus)
        corpus = re.sub(r'(, )+', ', ', corpus)
        return corpus

    def get_dset(dset_name):
        all_texts = []
        if dset_name == 'wsj':
            ds = get_wsj(ARGS.text_len*ARGS.n_texts)
            all_texts = [ds[i*ARGS.text_len:(i+1)*ARGS.text_len] for i in range(len(ds)//ARGS.text_len)]
            return all_texts
        for i in range(ARGS.n_texts):
            if dset_name.startswith('text'):
                topic = all_wiki_topics[i]
                lang = dset_name.split('-')[1]
                dset_text = get_wiki_text(topic, lang)
            elif dset_name == 'rand':
                with open('data/text/aurora-en.txt') as f:
                    text = f.read()[:ARGS.text_len]
                dset_text = ''.join(np.random.choice(list(set(text)), size=ARGS.text_len))
            elif dset_name == 'simp-en':
                dset_text = ''.join(np.random.choice(['boy','guy','her','his','the'], size=ARGS.text_len//3))
            elif dset_name == 'hps':
                with open('data/text/small-enwik9') as f:
                    dset_text = f.read()
            elif dset_name == 'hpm':
                with open('data/text/med-enwik9') as f:
                    dset_text = f.read()
            elif dset_name == 'repeat2':
                dset_text = rand_tiled(2, ARGS.text_len)
            elif dset_name == 'repeat5':
                dset_text = rand_tiled(5, ARGS.text_len)
            elif dset_name == 'repeat10':
                dset_text = rand_tiled(10, ARGS.text_len)
            all_texts.append(dset_text)

        return all_texts

    all_mean_results = {}
    all_std_results = {}
    os.makedirs('results/text', exist_ok=True)
    for d in dsets:
        text_dsets = get_dset(d)
        dset_results = []
        for td in text_dsets:
            new_results = discrete_lcc(td, ARGS.nits)
            dset_results.append(new_results)
        with open(f'results/text/{d}.json', 'w') as f:
            json.dump(dset_results, f)
        df = pd.DataFrame(dset_results)
        df.to_csv(f'results/text/{d}.csv')
        all_mean_results[d] = df.mean(axis=0)
        all_std_results[d] = df.std(axis=0)
    mean_results_df = pd.DataFrame(all_mean_results).T
    print(mean_results_df)
    if ARGS.dsets==['all'] and ARGS.n_texts==10:
        mean_results_df.to_csv('results/text/mean-results.csv')
        std_results_df = pd.DataFrame(all_std_results).T
        std_results_df.to_csv('results/text/std-results.csv')

