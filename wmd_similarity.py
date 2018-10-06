import pandas as pd
import time
import re
import codecs
from gensim.models import Word2Vec, KeyedVectors
from gensim.similarities import WmdSimilarity
from nltk import word_tokenize
import jieba

def LogInfo(stri):
    print(str(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))+'  '+stri)
    
def preprocess_data_en(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: document string
    Output: list of words
    '''     
    doc = doc.lower()
    doc = word_tokenize(doc)
    doc = [word for word in doc if word not in set(stopwords)]
    doc = [word for word in doc if word.isalpha()]
    return doc

def preprocess_data_cn(stopwords,doc):
    '''
    Function: preprocess data in Chinese including cleaning, tokenzing...
    Input: 
        stopwords: Chinese stopwords list
        doc: document string
    Output: list of words
    '''       
    # clean data
    doc = re.sub(u"[^\u4E00-\u9FFF]", "", doc) # delete all non-chinese characters
    doc = re.sub(u"[儿]", "", doc) # delete 儿
    # tokenize and move stopwords 
    doc = [word for word in jieba.cut(doc) if word not in set(stopwords)]   
    return doc

def wmd_similarity(lang,docs1,docs2):
    '''
    Input:
        lang: text language-Chinese for 'cn'/ English for 'en'
        docs1:  document strings list1
        docs2: document strings list2
    Output:
        WMD similarity list of docs1 and docs2 pairs
    '''
    
    # check if the number of documents matched
    assert len(docs1)==len(docs2) ,'Documents number is not matched!'
    assert len(docs1)!=0,'Documents list1 is null'
    assert len(docs2)!=0,'Documents list2 is null'
    assert lang=='cn' or lang=='en', 'Language setting is wrong'
    
    # change setting according to text language 
    if lang=='cn':
        model_path = '../model/cn.cbow.bin'
        stopwords_path = '../data/chinese_stopwords.txt'
        preprocess_data = preprocess_data_cn
    elif lang=='en':
        model_path = '../model/GoogleNews-vectors-negative300.bin'
        stopwords_path = '../data/english_stopwords.txt'
        preprocess_data = preprocess_data_en
        
    # load word2vec model  
    LogInfo('Load word2vec model...')
    model = KeyedVectors.load_word2vec_format(model_path,binary=True,unicode_errors='ignore')
    # normalize vectors
    model.init_sims(replace=True)
    
    # preprocess data
    stopwords= [w.strip() for w in codecs.open(stopwords_path, 'r',encoding='utf-8').readlines()]
    sims = []
    LogInfo('Calculating similarity...')
    for i in range(len(docs1)):        
        p1 = preprocess_data(stopwords,docs1[i])
        p2 = preprocess_data(stopwords,docs2[i])
        # calculate wmd similarity
        instance = WmdSimilarity(p1,model)
        sim = instance.get_similarities(p2)
        sims.append(sim[0])
    
    return sims

def example():
    # English text example
    docs1 = ['a speaker presents some products',
                 'vegetable is being sliced.',
                'man sitting using tool at a table in his home.']
    docs2 = ['the speaker is introducing the new products on a fair.',
                'someone is slicing a tomato with a knife on a cutting board.',
                'The president comes to China']
    sims = wmd_similarity('en',docs1,docs2)
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        
    # Chinese text example
    docs1 = ['做任何事都不能三天打鱼，两天晒网', 
             '学无止境', 
             '他整天愁眉苦脸']
    docs2 = ['对待每件事都不能轻言放弃', 
             '学海无涯，天道酬勤',
             '他和朋友去逛街']
    sims = wmd_similarity('cn',docs1,docs2)
    for i in range(len(sims)):
        print(docs1[i])
        print(docs2[i])
        print('Similarity: %.4f' %sims[i])
        
if __name__=='__main__':
    example()
