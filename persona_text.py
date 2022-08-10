#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 16:43:49 2021

@author: michael
"""

import warnings
import logging
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import os
import joblib
from sompy.sompy import SOMFactory

import random

from sklearnex import patch_sklearn
patch_sklearn()

#import multiprocessing
logging.getLogger('matplotlib.font_manager').disabled = True

warnings.filterwarnings("ignore")

csfont = {'fontname': 'DIN Condensed'}

colors = [(48/255, 61/255, 135/255), (217/255, 117/255, 169/255), (249/255, 249/255, 240/255), (241/255, 220/255, 169/255), (233/255, 189/255, 86/255), (225 /
                                                                                                                                                         255, 161/255, 71/255), (212/255, 143/255, 65/255), (186/255, 110/255, 54/255), (144/255, 78/255, 34/255), (125/255, 67/255, 28/255), (85/255, 45/255, 19/255)]
cmap_name = 'xenotheka'
cmap_name_r = 'xenotheka_r'
cm = LinearSegmentedColormap.from_list(cmap_name, colors, N=1000)
cm_r = LinearSegmentedColormap.from_list(
    cmap_name_r, np.flip(colors, axis=0), N=1000)


def corpus_vectors(corpus, dict_outpath=None, vectorizer='Count', vocab=None, max_features=20000, min_df=2, max_df=0.75):
    print('Vectorizing corpus of', len(corpus), 'documents')
    if vocab is not None:
        print('... using stored word list')
        vocab = vocab  # expecting a dictionary
    else:
        print('... calculating word list')
    if vectorizer == 'Tfidf':
        vectorizer = TfidfVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df, vocabulary=vocab, sublinear_tf=True)
        print('... using vectorizer: Tfidf')
    else:
        vectorizer = CountVectorizer(
            max_features=max_features, min_df=min_df, max_df=max_df, vocabulary=vocab)
        print('... using vectorizer: Count')
    print('... fitting to corpus')
    X = vectorizer.fit_transform(corpus)
    print('...', len(vectorizer.get_feature_names_out()), 'words found')
    if dict_outpath is not None:
        df_dict = pd.DataFrame(
            vectorizer.get_feature_names_out(), columns=['words'])
        print('... saving word list to dictionary')
        df_dict.to_csv(dict_outpath, sep=';', encoding='utf-8')
    else:
        pass
    words = vectorizer.get_feature_names_out()
    return X, words


csfont = {'fontname': 'DIN Condensed'}


def vectors_som(vectors, words, mapsize=[40, 60], normalization='var', initialization='pca', verbose=None, som_outpath=False, ask=True):
    # Check if SOM model is already stored at som_outpath
    if os.path.exists(som_outpath) == True:
        response = 'y'
        if ask:
            response = input("SOM model already saved, recalculate? [y/n] ")
        if response == 'y':
            print("... learning vector dimensionality (SOM)")
            som = SOMFactory().build(vectors, mapsize=[mapsize[0], mapsize[1]])
            # ,train_rough_len=100,train_finetune_len=200)
            som.train(n_job=1, verbose=verbose)

            topographic_error = som.calculate_topographic_error()
            quantization_error = som.calculate_quantization_error()
            print("Topographic error = %s\n Quantization error = %s" %
                  (topographic_error, quantization_error))
            joblib.dump(som, som_outpath)
            print("SOM model saved to:", som_outpath)
        if response == 'n':
            som = joblib.load(som_outpath)
    else:
        print("... learning vector dimensionality (SOM)")
        som = SOMFactory().build(vectors, mapsize=[mapsize[0], mapsize[1]])
        # ,train_rough_len=100,train_finetune_len=200)
        som.train(n_job=1, verbose=verbose)

        topographic_error = som.calculate_topographic_error()
        quantization_error = som.calculate_quantization_error()
        print("Topographic error = %s\n Quantization error = %s" %
              (topographic_error, quantization_error))
        joblib.dump(som, som_outpath)
        print("SOM model saved to:", som_outpath)

    print("... projecting codebook to vectors")
    bmus_2D = som.project_data(vectors)

    print("... gathering words by bmu")
    bmu_tmp = []
    for b in range(mapsize[0]*mapsize[1]):
        bmu_tmp.append(list(bmus_2D).count(b))
    words_by_bmu = pd.concat([words,
                              pd.Series(vectors.sum(axis=1), name='count')], axis=1)
    words_by_bmu['bmu'] = bmus_2D
    words_by_bmu.sort_values(by='count', ascending=False, inplace=True)
    bmu_cnt = []
    bmu_words = []
    for b in range(mapsize[0]*mapsize[1]):
        try:
            bmu_cnt.append(bmu_tmp[b])
            bmu_words.append(
                ' '.join([w for w in words_by_bmu[words_by_bmu['bmu'] == b]['words'].values]))
        except:
            bmu_cnt.append(0)
            bmu_words.append('')

    print("... returning codebook and matrix of words.")
    df_bmu_words = pd.DataFrame(bmu_words, columns=['words'])
    #df_bmu_words.to_csv(outpath + '/words_by_bmu.csv',encoding='utf-8',sep=';')
    return som.codebook.matrix.T, df_bmu_words


def plot_persona(codebook, bmu_wordmatrix, dir_path, mapsize=[40, 60], figsize=[80, 40], save_persona=False):
    plt.set_loglevel("info")
    cb = codebook
    bmu_words = bmu_wordmatrix.words
    print("... plotting collective persona")
    msz0, msz1 = mapsize[0], mapsize[1]
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    bmu_grid = cb.sum(axis=0).reshape(msz0, msz1)
    ax.imshow(bmu_grid, cmap=cm)
    words = bmu_words.values.reshape(msz0, msz1)
    n = 2  # number of words per line
    for i in range(msz0):
        for j in range(msz1):
            try:
                if bmu_grid[i, j] >= np.percentile(bmu_grid, 65) and bmu_grid[i, j] <= np.percentile(bmu_grid, 100):
                    color = 'w'
                else:
                    color = "w"
                ax.text(j, i, ' '.join([' '.join(g)+'\n' for g in [words[i, j].split()[k:k + n] for k in range(0, len(words[i, j].split()), n)]][:5]),
                        ha="center", va="center", color=color, **csfont, fontsize=4)
                ax.text(j-0.5, i-0.5, str(j)+'_'+str(i), ha="left",
                        va="top", color=color, **csfont, fontsize=4)
            except:
                pass
    plt.axis('off')
    plt.show()
    if save_persona == True:
        print('...saving collective persona.')
        path = dir_path + '/collective_persona.jpg'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        pass


def plot_persona_grid(dir_path, mapsize=[40, 60], figsize=[20, 40], save=True):
    plt.set_loglevel("info")
    cb = np.load(dir_path + '/som_codebook.npy')
    df = pd.read_csv(dir_path + '/doc_countvectors.csv',
                     encoding='utf-8', sep=';', usecols=['title', 'author'])
    print(df.shape[0])
    print("... plotting persona grid")
    msz0, msz1 = mapsize[0], mapsize[1]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     # creates 2x2 grid of axes
                     nrows_ncols=(int(df.shape[0]/5)+1, 5),
                     axes_pad=0.05,  # pad between axes in inch.
                     )

    for ax, im, t, a in zip(grid, cb, df['title'], df['author']):
        ax.text(30, 5, t+'\n ('+a+')', fontsize=8, **csfont, color='white',
                horizontalalignment='center', verticalalignment='center')
        ax.imshow(im.reshape(msz0, msz1), cmap=cm)
        ax.axis('off')
    plt.show()
    if save == True:
        print('... saving persona grid.')
        path = dir_path + '/persona_grid.jpg'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        pass


def plot_single_persona(corpus, codebook, bmu_wordmatrix, dir_path, author, title,
                        mapsize=[40, 60], figsize=[20, 30], save=True):
    msz0, msz1 = mapsize[0], mapsize[1]
    cb = codebook
    bmu_words = bmu_wordmatrix.words
    corpus = corpus.loc[:, ['title', 'author']]
    df_sel = pd.concat([corpus, pd.DataFrame(cb)], axis=1)
    df_sel = df_sel[df_sel.author.str.contains(
        author) & df_sel.title.str.contains(title)]
    print(df_sel.title.values[0], df_sel.author.values[0])
    csfont = {'fontname': 'DIN Condensed'}
    print('Plotting individual persona...')
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    bmu_grid = df_sel.iloc[:, 2:].values.reshape(msz0, msz1)
    ax.imshow(bmu_grid, cmap=cm)

    words = bmu_words.values.reshape(msz0, msz1)
    n = 2  # number of words per subgroup
    for i in range(msz0):
        for j in range(msz1):
            try:
                if bmu_grid[i, j] >= np.percentile(bmu_grid, 65) and bmu_grid[i, j] <= np.percentile(bmu_grid, 100):
                    color = 'w'
                else:
                    color = "w"
                ax.text(j, i, ' '.join([' '.join(g)+'\n' for g in [words[i, j].split()[k:k + n] for k in range(0, len(words[i, j].split()), n)]][:6]),
                        ha="center", va="center", color=color, **csfont, fontsize=4)
                ax.text(j-0.5, i-0.5, str(j)+'_'+str(i), ha="left",
                        va="top", color=color, **csfont, fontsize=4)
            except:
                pass
    ax.set_title(df_sel['title'].values[0] +
                 ' ('+df_sel['author'].values[0]+')', fontsize=16)
    plt.axis('off')
    plt.show()

    if save == True:
        print('... saving individual persona.')
        path = dir_path + '/doc_mask_' + \
            df_sel.title.values[0]+'_'+df_sel.author.values[0]+'.jpg'
        fig.savefig(path, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        pass


def plot_persona_talks(corpus, codebook, vectors, dir_path, tsne_path, replace_tsne=True, svd_nb=1000, perplexity=15, tsne_iter=5000, mapsize=[40, 60], figsize=(15, 15), highlight='', txt_color='white', persona_zoom=1, persona_max=10000, save=False):
    print("Calculating euclidean distance between vectors (t-SNE)")
    if replace_tsne == True:
        print("Computing TruncatedSVD")
        svd = TruncatedSVD(n_components=svd_nb)
        X_fit = svd.fit(vectors)
        print("Explained variance of data:",
              X_fit.explained_variance_ratio_.sum())
        X = svd.fit_transform(vectors)
        #X = vectors
        print("... recalculating t-SNE vectors")
        tsne_doc = TSNE(perplexity=perplexity, metric='euclidean', verbose=1,
                        n_iter=tsne_iter,
                        method='barnes_hut', n_components=2).fit_transform(X)
        print("... saving 2D t-SNE vectors")
        np.save(tsne_path, tsne_doc)
    else:
        tsne_doc = np.load(tsne_path)

    shape = [mapsize[1], mapsize[0]]
    cb = codebook

    print("... plotting persona talks")
    fig, ax = plt.subplots(figsize=(figsize[0], figsize[1]))
    artists = []
    if len(tsne_doc) > 10000:
        print("Large party, using a random sample of", persona_max, "masks.")
        sel = list(zip(tsne_doc, cb, corpus.title, corpus.author))
        random.shuffle(sel)
        #sel = zip(*sel)
        sel = sel[:persona_max]
    else:
        sel = zip(tsne_doc, cb, corpus.title, corpus.author)
    for xy, i, w, a in sel:
        x0, y0 = xy
        text = w+'\n(' + a + ')'
        if highlight in text:
            img = OffsetImage(
                i.reshape(shape[1], shape[0]), zoom=persona_zoom, cmap='gray', alpha=1.0)
        else:
            img = OffsetImage(
                i.reshape(shape[1], shape[0]), zoom=persona_zoom, cmap=cm, alpha=1.0)
        ab = AnnotationBbox(img, (x0, y0), xycoords='data', frameon=False)
        text = plt.text(x0, y0, w + '\n(' + a + ')', fontsize=4, **csfont,
                        color=txt_color, horizontalalignment='center', verticalalignment='bottom')
        artists.append(ax.add_artist(ab))
        artists.append(ax.add_artist(text))
    ax.update_datalim(tsne_doc)
    ax.autoscale()
    plt.axis('off')
    plt.show()

    if save == True:
        print("... saving persona grid.")
        outpath = dir_path + '/doc_friends_' + \
            str(shape[0]) + 'x' + str(shape[1]) + '.jpg'
        fig.savefig(fname=outpath, dpi=300, bbox_inches='tight', pad_inches=0)
    return tsne_doc


def plot_persona_talks_grid(df_namecols, tsne_vectors, codebook, outpath, mapsize=[10, 10], persona_shape=[40, 60],
                            figsize=(17, 17), save=False):
    print("... learning vector dimensionality (SOM)")
    som2 = SOMFactory().build(tsne_vectors, mapsize=mapsize)
    som2.train(n_job=8, shared_memory='yes', verbose=False)
    print("... projecting codebook to vectors")
    doc_2D = som2.project_data(tsne_vectors)
    df_tsneSOM = pd.concat(
        [pd.Series(doc_2D, name='bmu'), df_namecols, pd.DataFrame(codebook)], axis=1)
    bmu_cnt = []
    for i in range(mapsize[0]*mapsize[1]):
        bmu_cnt.append(np.count_nonzero(df_tsneSOM.bmu.values == i))
    max_bmu_cnt = np.array(bmu_cnt).max()
    print('... plotting figure')
    fig = plt.figure(figsize=figsize)
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     # creates 2x2 grid of axes
                     nrows_ncols=(mapsize[0], mapsize[1]),
                     axes_pad=0.00,  # pad between axes in inch.
                     )
    for i, ax in zip(range(mapsize[0]*mapsize[1]), grid):
        try:
            df_tsneSOM_sel = df_tsneSOM[df_tsneSOM['bmu'] == i].iloc[0]
            alpha = len(df_tsneSOM[df_tsneSOM['bmu'] == i])/max_bmu_cnt
            ax.text(30, 8, df_tsneSOM_sel['title']+'\n ('+df_tsneSOM_sel['author']+')', fontsize=4,
                    **csfont, color='white', horizontalalignment='center', verticalalignment='center')
            ax.imshow(df_tsneSOM_sel.iloc[3:].values.reshape(
                persona_shape[0], persona_shape[1]).astype('float'), cmap=cm, alpha=alpha)
            ax.axis('off')
        except:
            ax.axis('off')
        pass
    plt.show()
    if save is True:
        print('... saving figure.')
        fig.savefig(outpath, dpi=300, bbox_inches='tight', pad_inches=0)
    else:
        pass


def persona_talks_grid(cb_vectors, corpus, som_outpath, gridsize=[50, 50], som_verbose='info', plot_grid=False, save_plot=False, save_plot_path=False, masksize=[50, 50], figsize=(40, 40), ask=True):
    if os.path.exists(som_outpath) == True:
        response = 'y'
        if ask:
            response = input("SOM model already saved, recalculate? [y/n] ")
        if response == 'y':
            print("... learning vector dimensionality (SOM)")
            som = SOMFactory().build(cb_vectors, mapsize=[
                gridsize[0], gridsize[1]])
            som.train(verbose=som_verbose)
            print("Topographic error = %s\n Quantization error = %s" % (
                som.calculate_topographic_error(), som.calculate_quantization_error()))
            joblib.dump(som, som_outpath, compress=True)
            print("SOM model saved to:", som_outpath)
        if response == 'n':
            print("Loading saved SOM model")
            som = joblib.load(som_outpath)
    else:
        print("... learning vector dimensionality (SOM)")
        som = SOMFactory().build(cb_vectors, mapsize=[
            gridsize[0], gridsize[1]])
        som.train(verbose=som_verbose)
        print("Topographic error = %s\n Quantization error = %s" % (
            som.calculate_topographic_error(), som.calculate_quantization_error()))
        joblib.dump(som, som_outpath, compress=True)
        print("SOM model saved to:", som_outpath)

    print("... projecting codebook to vectors")
    cb_bmus = som.project_data(cb_vectors)
    df_cb_grid = pd.concat([pd.Series(cb_bmus, name='bmu'), corpus.loc[:, [
                           'title', 'author', 'idproprio']], pd.DataFrame(cb_vectors)], axis=1)

    if plot_grid == True:
        print('... plotting figure')
        fig = plt.figure(figsize=figsize)
        grid = ImageGrid(fig, 111,  # similar to subplot(111)
                         # creates 2x2 grid of axes
                         nrows_ncols=(gridsize[0], gridsize[1]),
                         axes_pad=0.00,  # pad between axes in inch.
                         )
        for i, ax in zip(range(gridsize[0]*gridsize[1]), grid):
            try:
                df_cb_grid_sel = df_cb_grid[df_cb_grid['bmu'] == i].iloc[0]
                ax.imshow(df_cb_grid_sel.iloc[3:].values.reshape(
                    masksize[0], masksize[1]).astype('float'), cmap=cm, alpha=1)
                ax.axis('off')
            except:
                ax.axis('off')
            pass
        plt.show()
        if save_plot == True:
            print('... saving figure.')
            fig.savefig(save_plot_path, dpi=300,
                        bbox_inches='tight', pad_inches=0)
        else:
            pass
    else:
        pass
    return df_cb_grid


def query_talks_grid(talks_grid, query_bmu, max_masks=10, figsize=(20, 40), masksize=[75, 75]):
    talks_grid_sel = talks_grid[talks_grid.bmu == query_bmu][:max_masks]
    fig = plt.figure(figsize=(figsize[0], figsize[1]))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     # creates 2x2 grid of axes
                     nrows_ncols=(int(talks_grid_sel.shape[0]/5)+1, 5),
                     axes_pad=0.05,  # pad between axes in inch.
                     )
    for ax, im, t, a in zip(grid, talks_grid_sel.iloc[:, 3:].values, talks_grid_sel.title, talks_grid_sel.author):
        ax.xaxis.set_label_position('top')
        ax.text(30, 5, t+'\n ('+a+')', fontsize=10, **csfont, color='white',
                horizontalalignment='center', verticalalignment='center')
        ax.imshow(im.reshape(masksize[0], masksize[1]), cmap=cm)
        ax.axis('off')
    plt.axis('off')
    plt.show()
