# NBClassifier_LinkedIn_feed
This project was created as a tool to filter specific types of info in my personal LinkedIn feed. This algorithm classifies a LinkedIn feed based on the post content, classifying the post with a 1 if it contains medicine/biotech-related topics, and a 0 if it does not. It then prints the name of the user that wrote a certain post (labeled with a 1), and the post itself. 
Naïve Bayes Classification 
This algorithm is based on Naïve Bayes (NB) classification, implemented through the TensorFlow probability package.  An NB classifier uses the Bayes rule to calculate the probability of each word occurring in each phrase, knowing the label of such phrase (0 or 1). In our case:
      ![alt text](https://github.com/gavrielhan/NBClassifier_Linkedin_feed/blob/d9e0d10d9030415da84eccec0d05618622f18645/Screenshot%202022-08-28%20193500.png)

When given a training set, the NB algorithm calculates the probability of each word in the data set to occur for a given label. Later, the same probability is used to evaluate the probability that a given phrase should be labeled as 0 or 1. This is done by multipathing the probability of each word in the tested phrase: 
           
![alt text](https://github.com/gavrielhan/NBClassifier_Linkedin_feed/blob/852ce4e9a972830cc4cf73a40e562697507f0959/Screenshot%202022-08-28%20194226.png)

All these probabilities are calculated by simply evaluating the frequency of each word, divided by the total words. The probability of having a certain label depends also form the frequency of those labels given in the training set. Just by looking at those calculations is clear that there are two main problems in such an analysis:

1) The order of the words and the meaning of the post are both irrelevant to the algorithm. If someone would have posted in my LinkedIn feed a bunch of nonsense words, all related to the medical/biotech field, the algorithm would have printed that as a suggested read for me. 

2) Strict dependency from the training data set, if a world is not present in the data set its calculated probability to receive a certain label will be 0, making the final probability for the phrase also 0, since all the probabilities are being multiplied by each other. This kind of mistake can be avoided by adding an arbitrary count to each word that the classifier encounters, avoiding a probability to be exactly 0.

The final decision on the label of one post is of course based on the highest probability between the two calculated. 
The two problems listed above are the main reasons this algorithm is defined as Naïve, more advanced classifiers take into account more parameters and can give more accurate results than this NB classifier. Though it still performs pretty well!

Note: This classifier is completely flexible, it can classify your LinkedIn feed by any topic, depending on the training set that is fed to it 
