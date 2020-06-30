# Sentiment-Analysis-for-Marathi
This program computes the sentiment score for the Marathi document. The system not only considered for positive/negative words but also considered words that increase or decrease the intensity of sentiment.
We also handled a negation handling issue, which is a challenging problem for the Marathi language. 

# NB_posmodel.sav
Naive Bayes Part of Sppech tagger model for the Marathi Language.

# sen_in.txt
Input is a Marathi language document for which polarity/sentiment is to be computed.
sen_out and senti_word are the output files generated.

# Sentiment Dataset
As a part of this project, we have built the followng sentiment Lexicon for the Marathi language.
Positive Dictionary: It contains 1695 words with the polarity of +1. E.g. vir, naayak, sardar etc.
Negative Dictionary: It contains 3720 words with the polarity of -1. E.g. bhepham, mandi, lach etc.
Inverse Dictionary: It contains 6 words. E.g. nahi, khoti, naste etc.
Increment Dictionary:  It contains words that increase the intensity of sentiment. It contains 22 words. E.g. khup, jast, atishay etc.
Decrement Dictionary:  It halves the polarity score of opinion word. E.g. kami, thoda etc.


