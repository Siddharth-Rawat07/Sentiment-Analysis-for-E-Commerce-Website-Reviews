from replacers import ExpandContractions 
from replacers import RepeatReplacer
from replacers import NormalizeWords
from replacers import JoinData
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from collections import defaultdict
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder


class DataCleaner(object):
    def cleanData(self,sentences):
        regExp = ExpandContractions()
        sentences = sentences.map(lambda x : regExp.expandContractions(x))
        replacer=RepeatReplacer()
        sentences = sentences.map(lambda x : replacer.replace(x))
        sentences = sentences.map(lambda x : x.encode('ascii', 'ignore'))
        normalize = NormalizeWords()
        sentences = sentences.map(lambda x : normalize.normalizeWords(x.decode("utf-8")))
        joinData=JoinData() 
        sentences = sentences.map(lambda x : joinData.join(x))
        ## reference:https://stackoverflow.com/questions/15586721/wordnet-lemmatization-and-pos-tagging-in-python
        # We are tagging the tokens with its respective POS using Lemmatizer
        tag_map = defaultdict(lambda : wn.NOUN)
        tag_map['J'] = wn.ADJ
        tag_map['V'] = wn.VERB
        tag_map['R'] = wn.ADV
        lmtzr = WordNetLemmatizer()
        for i in range(len(sentences)):
            pos_tokens=pos_tag(word_tokenize(sentences[i]))
            sentences[i] = [lmtzr.lemmatize(token,tag_map[tag[0]]) for (token,tag) in pos_tokens]
        sentences = sentences.map(lambda x : joinData.join(x))
        return sentences

class ModelAnalysis(object):
    # This method will vectorise and fit the data based on passed model.
    def VectAndTrans(self, sen_train,em_train,classifier, WithstopWords):
        vectorizer = TfidfVectorizer(ngram_range=(1,2),token_pattern=r'\b\w+\b',min_df=5, use_idf=True,stop_words=WithstopWords)
        vectorTrainedSent = vectorizer.fit_transform(sen_train)
        fittedClassfier = classifier.fit(vectorTrainedSent, em_train)
        return  fittedClassfier,vectorizer
    # This method will predict the transformed sentences
    def PredictEmotion(self,vectorizer, testsentences, classfier):
        testVect = vectorizer.transform(testsentences)
        return classfier.predict(testVect)
class PerformanceMetrices(object):
    #This method will give the accuracy from actual and predicted values
    def Accuracy(self,actualValues, predictedValues):
        return accuracy_score(actualValues,predictedValues)
    def Confusion_matrix(self, actualValues , predictedValues):
        return confusion_matrix(actualValues, predictedValues)
class WordLength(object):
    def getMaxWordLength(self, sentences):
        maxlenOfWords = 0
        for sentence in sentences:
            if len(sentence.split())> maxlenOfWords:
                maxlenOfWords = len(sentence.split())
        return maxlenOfWords
class OneHotEncoding(object):
    def GetOneHotEncodedMatrix(self, values):
        labelEncoder = LabelEncoder()
        labelEncoded = labelEncoder.fit_transform(values)
        onehotEncode = OneHotEncoder()
        return onehotEncode.fit_transform(labelEncoded.reshape(-1,1))