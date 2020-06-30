import nltk
from nltk.corpus import indian
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os
import yaml
pos_count = 0
neg_count = 0
inc_count = 0
dec_count = 0
inv_count = 0
aflag = 0
class Splitter(object):
    def __init__(self):
        self.nltk_splitter = nltk.data.load('tokenizers/punkt/english.pickle')
        self.nltk_tokenizer = nltk.tokenize.TreebankWordTokenizer()
        
    def split(self, text):
        """
	input format: a paragraph of text
	output format: a list of lists of words.
	e.g.: [['this', 'is', 'a', 'sentence'], ['this', 'is', 'another', 'one']]
	"""
        sentences = self.nltk_splitter.tokenize(text)
        tokenized_sentences = [self.nltk_tokenizer.tokenize(sent) for sent in sentences]
        return tokenized_sentences
def value_of(sentiment):
    #print("pos_count=",pos_count)
    global pos_count
    global neg_count
    if sentiment == 'positive':
        pos_count += 1
        return 1
    if sentiment == 'negative':
        neg_count += 1
        return -1
    return 0


def sentence_score(sentence_tokens, previous_token, acum_score,invflag,aflag):
    global inc_count
    global dec_count
    global inv_count
    
   
    if not sentence_tokens:
        if invflag==1:
            return (acum_score*-1)
        else:
            return acum_score
    else:
        current_token = sentence_tokens[0]
        if (len(sentence_tokens) > 1):
            next_token = sentence_tokens[1]
        else:
           
            next_token = -1
        
       # print("next token", sentence_tokens[1][2])
        #if(len(sentence_tokens)!=1):
         #   next_token = sentence_tokens[1]
          #  print("next token", next_token[2])
           # flag=1
        #print("length of current token=",len(current_token[2]),current_token[2][0])
        if previous_token is None:
           # acum_score=0
            if(len(current_token[2]) == 2 and (current_token[0] == 'आनंद' or current_token[0] =='कुशल')):
                print("current token",current_token[0])
                token_score = 0
            elif(len(current_token[2])==2 and current_token[2][1].startswith('NN')== True and current_token[0] == 'शूर'):
                tags = current_token[2]
                token_score = sum([value_of(tag) for tag in tags])
            elif(len(current_token[2])==2 and current_token[2][1].startswith('NN')== False): 
                tags = current_token[2]
                token_score = sum([value_of(tag) for tag in tags])
            elif(len(current_token[2])==3 and current_token[2][2].startswith('NN')== False):
                if next_token ==-1:
                    print("next token")
                    tags = current_token[2]
                    token_score = sum([value_of(tag) for tag in tags])
                else:
                    
                    token_score=0
            else:
               
                token_score=0
            acum_score=token_score
        else:
            
            tags = current_token[2]
            token_score = sum([value_of(tag) for tag in tags])
   #         print("token score",token_score)
            
        if previous_token is not None:
            previous_tags = previous_token[2]
          
                
            if 'inc' in previous_tags:
                inc_count += 1               
                token_score *= 2.0
                #acum_score += token_score
            if 'dec' in previous_tags:
                token_score /= 2.0
                dec_count += 1
                #acum_score += token_score
            if 'inv' in previous_tags:
                inv_count += 1
                if token_score!=0 and aflag==0:
  #                  print("token_score in previous", token_score)
                    token_score *= -1.0
                    invflag=0
                elif aflag==1:
                    invflag=0
                else:
                    invflag=1
           
                
            
                #acum_score += token_score
            acum_score += token_score
    #        print("total acum score",acum_score)
                
        if(next_token!=-1):
                #next_token = sentence_tokens[1]
                #print("next token", next_token[2])
           

            if 'inv' in next_token[2][0]:
                if invflag == 1 and acum_score!=0:
                    acum_score *= -1.0
                    invflag=0
     #           print("token_score in next", acum_score)
      #          print("inverse")
                inv_count += 1
                if acum_score!=0:   
                    acum_score *= -1.0
                    #invflag=0
       #             print("token_score in next", acum_score)
                    
                    aflag=1
                else:
                    aflag=0
                                #acum_score *= -1.0
              
        return sentence_score(sentence_tokens[1:], current_token, acum_score,invflag,aflag)
def sentiment_score(review):
    return sum([sentence_score(sentence, None, 0.0,0,0) for sentence in review])




class DictionaryTagger(object):
    def __init__(self, dictionary_paths):
        files = [open(path, 'r',encoding='utf8') for path in dictionary_paths]
        dictionaries = [yaml.load(dict_file) for dict_file in files]
        map(lambda x: x.close(), files)
        self.dictionary = {}
        self.max_key_size = 0
        for curr_dict in dictionaries:
            for key in curr_dict:
                if key in self.dictionary:
                    self.dictionary[key].extend(curr_dict[key])
                else:
                    self.dictionary[key] = curr_dict[key]
                    self.max_key_size = max(self.max_key_size, len(key))
                    
    def tag(self, postagged_sentences):
        return [self.tag_sentence(sentence) for sentence in postagged_sentences]

    def tag_sentence(self, sentence, tag_with_lemmas=False):
        """
	the result is only one tagging of all the possible ones.
	The resulting tagging is determined by these two priority rules:
	- longest matches have higher priority
	- search is made from left to right
	"""
        tag_sentence = []
        N = len(sentence)
        if self.max_key_size == 0:
            self.max_key_size = N
        i = 0
        while (i < N):
            j = min(i + self.max_key_size, N) #avoid overflow
            tagged = False
            while (j > i):
                expression_form = ' '.join([word[0] for word in sentence[i:j]])
                expression_lemma = ' '.join([word[1] for word in sentence[i:j]])
                if tag_with_lemmas:
                    literal = expression_lemma
                   
                else:
                    literal = expression_form 
                l=len(literal)    
                if literal in self.dictionary:
                    #self.logger.debug("found: %s" % literal)
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal]]
                   
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                    senti=(expression_form,taggings[0])
                    fout1.write(str(senti))
                elif literal[:l-1] in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal[:l-1]]]
                   
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                    senti=(expression_form,taggings[0])
                    fout1.write(str(senti))
                elif literal[:l-2] in self.dictionary:
                    is_single_token = j - i == 1
                    original_position = i
                    i = j
                    taggings = [tag for tag in self.dictionary[literal[:l-2]]]
                  
                    tagged_expression = (expression_form, expression_lemma, taggings)
                    if is_single_token: #if the tagged literal is a single token, conserve its previous taggings:
                        original_token_tagging = sentence[original_position][2]
                        tagged_expression[2].extend(original_token_tagging)
                    tag_sentence.append(tagged_expression)
                    tagged = True
                    senti=(expression_form,taggings[0])
                    fout1.write(str(senti))
                
                else:
                    j = j - 1
            if not tagged:
                tag_sentence.append(sentence[i])
                i += 1
        return tag_sentence
#mpos = indian.tagged_sents('marathi_pos_rad_3NOV17.pos')
def pos_features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    features = {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'prefix-1': sentence[index][:0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1:],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
      
    }
    if index == 0:
        features["prev-word"] = "<START>"
    else:
        features["prev-word"] = sentence[index-1]
    return features

splitter = Splitter()
fin=open("sen_in.txt","r",encoding='utf8')
word_to_be_tagged=fin.read()
fout=open("sen_out.txt","w",encoding='utf8')
fout1=open("senti_words.txt","w",encoding='utf8')
splitted_sentences = splitter.split(word_to_be_tagged)
print(splitted_sentences)

b=nltk.word_tokenize(word_to_be_tagged)

a=nltk.word_tokenize(word_to_be_tagged)
filename = 'NN_posmodel.sav'
tree_model = pickle.load(open(filename, 'rb'))

sc=0

dicttagger = DictionaryTagger(['positive_m.yml','dec_m.yml','negative_m.yml','inc_m.yml','inv_m.yml'])
for sen in splitted_sentences:
    print("sentence=",sen)
    f1=[]
    s=[]
    sent=[]
    for i, word in enumerate(sen):
      sent.append(word)
    print("sen=",sent)
    for i, word in enumerate(sent):   
      f1.append((pos_features(sent,i)))
    sout="sentiment words for  "+str(sen)+"="
  
    y_pred = tree_model.predict(f1)
    fout1.write(sout)
    ts=[]
    a=[w for w in sent]
    for i in range (len(y_pred)):
        
  
        ts.append((a[i],y_pred[i]))
  
    for (word,postag) in ts:
        print((word,postag))
    pos = [[(word, word, [postag]) for (word, postag) in ts]] 
    print("dict_tagged_data")
    print(pos)

    dict_tagged_sentences = dicttagger.tag(pos)
    print("dict tagged sentences=")
    print(dict_tagged_sentences)
    print("sentiment score of ", sen)
    
    sc1=sentiment_score(dict_tagged_sentences)
    sout="sentiment score of "+str(sen)+"="+str(sc1)
    print(sout)
    fout.write(sout)
    fout.write("\n")
    #print(sc1)
    sc=sc+sc1
    fout1.write("\n")
        



print("Total score = ",sc)

fout.write(str("Total score = "+str(sc)))
names = ["Positive","Negative","Increment","Decrement","Inverse"]
n=len(names)
counts = np.array([pos_count,neg_count,inc_count,dec_count,inv_count])
ind = np.arange(n)
width = 0.35
print(n)
print(ind)

plt.bar(ind, counts, width, color="red")

plt.ylabel("Word Count--->")
plt.title("Sentiment words count")
plt.xticks(ind+width/2, ("Positive","Negative","Increment","Decrement","Inverse"))
#plt.show()
plt.savefig("sentiment.png")

plt.show()

fin.close()
fout.close()
fout1.close()




