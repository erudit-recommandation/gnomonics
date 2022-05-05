#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 20 17:29:29 2020

@author: michael
"""
import string
from nltk import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import gensim
import spacy
from bs4 import BeautifulSoup

def preprocessing_text(text,language,remove_sw=True,stem=False,min_word_len=3):
    text_1 = ' '.join(word_tokenize(text.lower().strip(),language=language))
    text_1 = text_1.replace('-',' ').replace('\'',' ')
    #print(text_1,'\n')
    punctuation = string.punctuation+'’'+'«'+'»'+'—'+'•'+'■'
    text_p = "".join([char for char in text_1 if char not in punctuation])
    #print(text_p,'\n')
    text_f = ''.join([i for i in text_p if not i.isdigit()])
    text_f = ' '.join([w for w in text_f.split() if len(w) > min_word_len])
    if remove_sw is True:
        if language == 'french':
            stop_word_string = "a abord absolument afin ah ai aie aient aies ailleurs ainsi ait allaient allo allons allô alors anterieur anterieure anterieures apres après as assez attendu au aucun aucune aucuns aujourd aujourd'hui aupres auquel aura aurai auraient aurais aurait auras aurez auriez aurions aurons auront aussi autant autre autrefois autrement autres autrui aux auxquelles auxquels avaient avais avait avant avec avez aviez avions avoir avons ayant ayez ayons b bah bas basee bat beau beaucoup bien bigre bon boum bravo brrr c car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui celui-ci celui-là celà cent cependant certain certaine certaines certains certes ces cet cette ceux ceux-ci ceux-là chacun chacune chaque cher chers chez chiche chut chère chères ci cinq cinquantaine cinquante cinquantième cinquième clac clic combien comme comment comparable comparables compris concernant contre couic crac d da dans de debout dedans dehors deja delà depuis dernier derniere derriere derrière des desormais desquelles desquels dessous dessus deux deuxième deuxièmement devant devers devra devrait different differentes differents différent différente différentes différents dire directe directement dit dite dits divers diverse diverses dix dix-huit dix-neuf dix-sept dixième doit doivent donc dont dos douze douzième dring droite du duquel durant dès début désormais e effet egale egalement egales eh elle elle-même elles elles-mêmes en encore enfin entre envers environ es essai est et etant etc etre eu eue eues euh eurent eus eusse eussent eusses eussiez eussions eut eux eux-mêmes exactement excepté extenso exterieur eûmes eût eûtes f fais faisaient faisant fait faites façon feront fi flac floc fois font force furent fus fusse fussent fusses fussiez fussions fut fûmes fût fûtes g gens h ha haut hein hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hum hurrah hé hélas i ici il ils importe j je jusqu jusque juste k l la laisser laquelle las le lequel les lesquelles lesquels leur leurs longtemps lors lorsque lui lui-meme lui-même là lès m ma maint maintenant mais malgre malgré maximale me meme memes merci mes mien mienne miennes miens mille mince mine minimale moi moi-meme moi-même moindres moins mon mot moyennant multiple multiples même mêmes n na naturel naturelle naturelles ne neanmoins necessaire necessairement neuf neuvième ni nombreuses nombreux nommés non nos notamment notre nous nous-mêmes nouveau nouveaux nul néanmoins nôtre nôtres o oh ohé ollé olé on ont onze onzième ore ou ouf ouias oust ouste outre ouvert ouverte ouverts o| où p paf pan par parce parfois parle parlent parler parmi parole parseme partant particulier particulière particulièrement pas passé pendant pense permet personne personnes peu peut peuvent peux pff pfft pfut pif pire pièce plein plouf plupart plus plusieurs plutôt possessif possessifs possible possibles pouah pour pourquoi pourrais pourrait pouvait prealable precisement premier première premièrement pres probable probante procedant proche près psitt pu puis puisque pur pure q qu quand quant quant-à-soi quanta quarante quatorze quatre quatre-vingt quatrième quatrièmement que quel quelconque quelle quelles quelqu'un quelque quelques quels qui quiconque quinze quoi quoique r rare rarement rares relative relativement remarquable rend rendre restant reste restent restrictif retour revoici revoilà rien s sa sacrebleu sait sans sapristi sauf se sein seize selon semblable semblaient semble semblent sent sept septième sera serai seraient serais serait seras serez seriez serions serons seront ses seul seule seulement si sien sienne siennes siens sinon six sixième soi soi-même soient sois soit soixante sommes son sont sous souvent soyez soyons specifique specifiques speculatif stop strictement subtiles suffisant suffisante suffit suis suit suivant suivante suivantes suivants suivre sujet superpose sur surtout t ta tac tandis tant tardive te tel telle tellement telles tels tenant tend tenir tente tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous tout toute toutefois toutes treize trente tres trois troisième troisièmement trop très tsoin tsouin tu té u un une unes uniformement unique uniques uns v va vais valeur vas vers via vif vifs vingt vivat vive vives vlan voici voie voient voilà voire vont vos votre vous vous-mêmes vu vé vôtre vôtres w x y z zut à â ça ès étaient étais était étant état étiez étions été étée étées étés êtes être ô "
            stop_words = stop_word_string.split(' ')
        else:
            stop_words = stopwords.words(language)
        words = text_f.split()
        text_f = ' '.join([word for word in words if word not in stop_words])
    else:
        pass
    #print(text_f,'\n')
    if stem is True:
        words = text_f.split()
        text_f = ' '.join([SnowballStemmer(language=language,ignore_stopwords=True).stem(w) for w in words])
    else:
        pass
    #print(text_f,'\n')
    return text_f

def sent_to_words(sentences):
    for sentence in sentences:
        yield gensim.utils.simple_preprocess(str(sentence), deacc=True)  # deacc=True removes punctuations
           
def lemmatizing_text(text,language,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV', 'PROPN'],min_word_len=3):
    data = text
    #data = text.values.tolist()
    data_words = list(sent_to_words(data))
    if language=='french':
        nlp = spacy.load('fr_core_news_sm',disable=['ner'])
    else:
        nlp = spacy.load('en_core_web_sm',disable=['parser','ner'])
    nlp.max_length = 50000000
    if language == 'french':
        stop_word_string = "a abord absolument afin ah ai aie aient aies ailleurs ainsi ait allaient allo allons allô alors anterieur anterieure anterieures apres après as assez attendu au aucun aucune aucuns aujourd aujourd'hui aupres auquel aura aurai auraient aurais aurait auras aurez auriez aurions aurons auront aussi autant autre autrefois autrement autres autrui aux auxquelles auxquels avaient avais avait avant avec avez aviez avions avoir avons ayant ayez ayons b bah bas basee bat beau beaucoup bien bigre bon boum bravo brrr c car ce ceci cela celle celle-ci celle-là celles celles-ci celles-là celui celui-ci celui-là celà cent cependant certain certaine certaines certains certes ces cet cette ceux ceux-ci ceux-là chacun chacune chaque cher chers chez chiche chut chère chères ci cinq cinquantaine cinquante cinquantième cinquième clac clic combien comme comment comparable comparables compris concernant contre couic crac d da dans de debout dedans dehors deja delà depuis dernier derniere derriere derrière des desormais desquelles desquels dessous dessus deux deuxième deuxièmement devant devers devra devrait different differentes differents différent différente différentes différents dire directe directement dit dite dits divers diverse diverses dix dix-huit dix-neuf dix-sept dixième doit doivent donc dont dos douze douzième dring droite du duquel durant dès début désormais e effet egale egalement egales eh elle elle-même elles elles-mêmes en encore enfin entre envers environ es essai est et etant etc etre eu eue eues euh eurent eus eusse eussent eusses eussiez eussions eut eux eux-mêmes exactement excepté extenso exterieur eûmes eût eûtes f fais faisaient faisant fait faites façon feront fi flac floc fois font force furent fus fusse fussent fusses fussiez fussions fut fûmes fût fûtes g gens h ha haut hein hem hep hi ho holà hop hormis hors hou houp hue hui huit huitième hum hurrah hé hélas i ici il ils importe j je jusqu jusque juste k l la laisser laquelle las le lequel les lesquelles lesquels leur leurs longtemps lors lorsque lui lui-meme lui-même là lès m ma maint maintenant mais malgre malgré maximale me meme memes merci mes mien mienne miennes miens mille mince mine minimale moi moi-meme moi-même moindres moins mon mot moyennant multiple multiples même mêmes n na naturel naturelle naturelles ne neanmoins necessaire necessairement neuf neuvième ni nombreuses nombreux nommés non nos notamment notre nous nous-mêmes nouveau nouveaux nul néanmoins nôtre nôtres o oh ohé ollé olé on ont onze onzième ore ou ouf ouias oust ouste outre ouvert ouverte ouverts o| où p paf pan par parce parfois parle parlent parler parmi parole parseme partant particulier particulière particulièrement pas passé pendant pense permet personne personnes peu peut peuvent peux pff pfft pfut pif pire pièce plein plouf plupart plus plusieurs plutôt possessif possessifs possible possibles pouah pour pourquoi pourrais pourrait pouvait prealable precisement premier première premièrement pres probable probante procedant proche près psitt pu puis puisque pur pure q qu quand quant quant-à-soi quanta quarante quatorze quatre quatre-vingt quatrième quatrièmement que quel quelconque quelle quelles quell quelqu'un quelque quelques quels qui quiconque quinze quoi quoique r rare rarement rares relative relativement remarquable rend rendre restant reste restent restrictif retour revoici revoilà rien s sa sacrebleu sait sans sapristi sauf se sein seize selon semblable semblaient semble semblent sent sept septième sera serai seraient serais serait seras serez seriez serions serons seront ses seul seule seulement si sien sienne siennes siens sinon six sixième soi soi-même soient sois soit soixante sommes son sont sous souvent soyez soyons specifique specifiques speculatif stop strictement subtiles suffisant suffisante suffit suis suit suivant suivante suivantes suivants suivre sujet superpose sur surtout t ta tac tandis tant tardive te tel telle tellement telles tels tenant tend tenir tente tes tic tien tienne tiennes tiens toc toi toi-même ton touchant toujours tous tout toute toutefois toutes treize trente tres trois troisième troisièmement trop très tsoin tsouin tu té u un une unes uniformement unique uniques uns v va vais valeur vas vers via vif vifs vingt vivat vive vives vlan voici voie voient voilà voire vont vos votre vous vous-mêmes vu vé vôtre vôtres w x y z zut à â ça ès étaient étais était étant état étiez étions été étée étées étés êtes être ô "
        stop_words = stop_word_string.split(' ')
        stop_word_string = ' '.join(stopwords.words('english')) + ' ' + stop_word_string
        stop_words = list(set(stop_word_string.split()))
    else:
        stop_words = stopwords.words(language)
    texts_out = []
    for sent in data_words:
        doc = nlp(" ".join(sent)) 
        doc = [token.lemma_ for token in doc if token.pos_ in allowed_postags]
        doc = [w for w in doc if len(w) >= 3]
        texts_out.append(' '.join([word for word in doc if word not in stop_words]))
    return texts_out
    
def lemmatizing_text_old(text,language,allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    # this is performed on a series of texts that have been tokenized.
    data = text.values.tolist()
    data_words = list(sent_to_words(data))
    bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) # higher threshold fewer phrases.
    trigram = gensim.models.Phrases(bigram[data_words], threshold=100) 
    bigram_mod = gensim.models.phrases.Phraser(bigram)
    trigram_mod = gensim.models.phrases.Phraser(trigram)
    #data_words_bigrams = [bigram_mod[doc] for doc in data_words]
    data_words_trigrams = [trigram_mod[bigram_mod[doc]] for doc in data_words]
    if language=='french':
        nlp = spacy.load('fr_core_news_sm', disable=['parser', 'ner'])
        #nlp = spacy.load('fr_dep_news_trf', disable=['parser', 'ner'])
    else:
        nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nlp.max_length = 50000000
    texts_out = []
    for sent in data_words_trigrams:
        doc = nlp(" ".join(sent)) 
        texts_out.append(' '.join([token.lemma_ for token in doc if token.pos_ in allowed_postags]))
    return texts_out

def most_frequent(List): 
    counter = 0
    num = List[0] 
      
    for i in List: 
        curr_frequency = List.count(i) 
        if(curr_frequency> counter): 
            counter = curr_frequency 
            num = i 
  
    return num

def preprocessing_html(file,font_size=False):
    with open(file,'r') as f:
        contents=f.read()
        soup = BeautifulSoup(contents, 'lxml')

    fs = soup.find_all(name='p')
    print(len(fs))

    if font_size is False:
        List = []
        span = [s for s in soup.find_all(name='span') if len(s.text.split(' ')) > 30]
        for f in range(len(span)):
            try:
                List.append(span[f]['class'][0])
            except:
                pass
        font = most_frequent(List)
        print(font)
    else:
        print(font_size)
        font = font_size
    
    para = []
    for p in range(len(fs)):
        span = []
        fs_s = fs[p].find_all('span')
        for s in range(len(fs_s)):
            if fs_s[s]['class'][0] == font:
                span.append(fs_s[s].text)
            else:
                pass
        if len(span)>0:
            para.append(' '.join(span))
        else:
            pass

    f = open(file.split('.')[0]+'.txt', "w")
    f.write('\n\n'.join(para))