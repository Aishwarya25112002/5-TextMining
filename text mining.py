# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 08:32:01 2023

@author: hp
"""

#               TExt_MINING

sentence="we are learning TextMining from sanjivani AI"

#IF we want to knoe the position of learning

sentence.index("learning")
#it will show learning at position 7
#this is going to show the charater position from 0 in cluding
##############################################
#finding index of word
#we want to know the pposition of textmining word
sentence.split().index("TextMining")
#it will split the wordsin list and count the position
#if you want to see the list select senetence.split()
#and it will show at 3

##################################
#suppose we wantprint any in word reverse word
sentence.split()[2][::-1]
#start:end:-1(start) will start from -1,-2,-3 till the end
#learning will be printed gninrael

#########################################################

#suppose we want to print first and last word of sentence

words=sentence.split()
first_word=words[0]
first_word
last_word=words[-1]
last_word

###################################################

#now we want to concate the first and last word

concat_word=first_word+" "+last_word
concat_word
###########################
#we want to print even words from sentence

[words[i] for i in range(len(words)) if i%2==0]

###########################3
#we want to display AI word

sentence[-3]

############################
#we want to display sentence in reverse order
sentence[::-1]
#here sentence get reversed 
##################################
#IMP=R.R
#suppose we want to select each word and print in reversed order

words
print(" ".join(word[::-1]for word in words))
#here words get reversed not sentence
###################################################3

#               Tokenization

import nltk
nltk.download('punkt')
from nltk import word_tokenize
w=word_tokenize("I am reading NLP Fundamentals")
print(w)
############################
#part of speech (pos)tagging
nltk.download('averaged_perception_tagger')
nltk.pos_tag(w)
#it is going to mention parts of speech
##############################

#stop word from nltk libarary
from nltk.corpus import stopwords
stop_words=stopwords.words('English')
#you can verify 179 stop words in variable explorer
print(stop_words)

sentence1="I am learning NLP:it is one of the most popular library"
#first we will tokenize the sentence
sentence_words=word_tokenize(sentence1)
print(sentence_words)

#now let us filter the sentence1 using stop_words
sentence_no_stops=" ".join([words for words in sentence_words])
print(sentence_no_stops)
sentence1

#you can notice that am,is,of,most,popular,in are missing

#################################
#suppose we want to replace words in string

sentence2="I visited MY from IND on 14-02-19"
normalized_sentence=sentence2.replace("MY","Malaysia").replace("IND","India")
normalized_sentence=normalized_sentence.replace("-19","-2020")
print(normalized_sentence)

####################################
#support we want auto correction in the sentence

from autocorrect import Speller
#delcalare the function speller defined for English

spell=Speller(lang='en')
spell("English")
########################
#suppose we want to correct whole sentence

sentence3="Natural lanagage processin deals withh the aart of extracting sentiiiments"
sentence3=word_tokenize(sentence3)
corrected_sentence=" ".join([spell(word)for word in sentence3])
print(corrected_sentence)

