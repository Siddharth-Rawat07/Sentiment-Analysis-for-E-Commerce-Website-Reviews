import re
import pandas as pd
import re 
import numpy as np
from sklearn.feature_extraction import text
from nltk.corpus import stopwords
from nltk.corpus import wordnet
#Reference : https://gist.github.com/nealrs/96342d8231b75cf4bb82#gistcomment-2182197
# Words like won't, can't are expanded to will not, cannot,..
class ExpandContractions():
    cList = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it had",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so is",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there had",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we had",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn'twouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'alls": "you alls",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you had",
    "you'd've": "you would have",
    "you'll": "you you will",
    "you'll've": "you you will have",
    "you're": "you are",
    "you've": "you have"
}
# Thier might be a possibility that words like you'll've might get splitted
# into you will've. Hence, we are sorting the data based on the apostrophe and then expand
# them.
    df_cList = pd.DataFrame(list(cList.items()), columns = ['contraction', 'expansion'])
    df_cList['apos_count'] = df_cList['contraction'].apply(lambda x: x.count("'"))
    df_cList.sort_values('apos_count', ascending=False, inplace=True)
    c_re = re.compile('(%s)' % '|'.join(df_cList["contraction"]))

    def expandContractions(self,text, c_re=c_re,df_cList=df_cList):
        def replace(match):
            return df_cList[df_cList['contraction']==match.group(0)]['expansion'].item() 
        return c_re.sub(replace, text.lower())


"""
Created on Mon Apr  8 14:44:27 2019

@author: Anvitha
"""

import re 
from nltk.corpus import wordnet
# The class will replace the words like happyyyyyy into happy by
# by checking the word in synset.
class RepeatReplacer(object):    
    def __init__(self):        
        self.repeat_regexp = re.compile(r"(.)\1{2,}")        
        self.repl = r'\1'    
    def replace(self, word):        
        if wordnet.synsets(word):            
            return word        
        repl_word = self.repeat_regexp.sub(self.repl, word)        
        if repl_word != word:            
            return self.replace(repl_word)        
        else:            
            return repl_word

# This class will form the sentence from tokens.
class JoinData(object):
    def join(self, words):
        return ' '.join(words)

# This class will convert the lower case and allow only unicode characters by
# removing emojis.
class NormalizeWords(object):
    def normalizeWords(self,text):
        return re.compile(r'\W+', re.UNICODE).split(text.lower())

# This class will get the stops words and removes not from them which is 
# necessary for conveying negative emotions.
class GetStopWords(object):
    def getStopWords(self):
        setOfWords = []
        newstopwords = (stopwords.words('english'))
        newstopwords.remove('not')
        for word in text.ENGLISH_STOP_WORDS:
            if(word != 'not'):
                setOfWords.append(word)
        return (set(newstopwords) | set(setOfWords))
        