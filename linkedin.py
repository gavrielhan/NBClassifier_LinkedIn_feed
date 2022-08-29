import tensorflow_hub as hub
import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import getpass
from bot_studio import *
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

def get_post_content(data):
    my_feed=[]
    post_author = []
    for dic in data:
        my_feed.append(dic['Post Content'])
        post_author.append(dic['User Name'])
        if len(my_feed)>1:
            if my_feed[len(my_feed)-1]=="Show more Feed Updates":
                my_feed.pop(len(my_feed)-1)
                post_author.pop(len(post_author)-1)
            if len(my_feed) > 1:
                if my_feed[len(my_feed)-1]==my_feed[len(my_feed)-2]:
                    my_feed.pop(len(my_feed)-1)
                    post_author.pop(len(post_author) - 1)
                if len(my_feed)>1:
                    if my_feed[len(my_feed)-1]=="":
                        my_feed.pop(len(my_feed) - 1)
                        post_author.pop(len(post_author) - 1)

    return my_feed, post_author

def find_main_post(my_feed): #When receiving the feed you will get a lot of other useless info in the same string as the main post
    for i in range(len(my_feed)):
        sep_feed = my_feed[i].split("\n")
        len_feed=[]
        for phrase in sep_feed:
            len_feed.append(len(phrase))
        ind = len_feed.index(max(len_feed))  #get the biggest string, assuming that that string is going to be the post
        my_feed[i] = sep_feed[ind]
    return my_feed

def process_feed_data(data):
    my_feed,author = get_post_content(data)
    my_feed = find_main_post(my_feed)
    return my_feed,author

def labelling_machine(my_feed): #if you wat to label the feed yourself to then compare it with the result
    label_list =[]
    for post in my_feed:
        print(post)
        label = int(input("Please input a label for the post. The label can be either 0 or 1: "))
        label_list.append(label)
    return label_list



linkedin=bot_studio.linkedin()
username = input("Please insert your Linkedin username: ")
password = getpass.getpass("Please insert your Linkedin password: ")
n = int(input("Please insert the desired number of scrolls on your LinkedIn feed: "))
linkedin.login(username=username, password=password)
os.chdir("C:\\Users\\gavri\\Desktop\\NBclassifier")


data=[]
k=0
while(k<n):
    response=linkedin.get_feed()
    for key in response['body']:
        data.append(key)
    linkedin.scroll()
    k=k+1
    print("Currently at iteration:",k)


my_feed,post_author = process_feed_data(data)
#eliminate hebrew from feed
i=0
while i < len(my_feed):
    if chr(1488) in my_feed[i]:
         my_feed.pop(i)
         post_author.pop(i)
         i=i-1
    i+=1


test = pd.DataFrame(my_feed)

embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/3")

train_set = pd.read_csv('train_set_NBclass.csv') #upload the training set
del train_set['Unnamed: 0']
np_list = np.asarray(train_set['0'].tolist())
my_feed_train = tf.convert_to_tensor(np_list) #convert it to a tensor
x_train = embed(my_feed_train)
x_train_matrix = x_train['outputs'].numpy() #preprare the training matrix to add to the classifier
y_train = tf.constant(train_set['labels'])
# get 1 if the post is about pharma/medicine/biotech, 0 if is not


#label_byhand = labelling_machine(my_feed) #if you whish to check the predictions



np_list = np.asarray(test[0].tolist())
my_feed_test = tf.convert_to_tensor(np_list)
X_test_embeddings = embed(my_feed_test)
X_test_matrix = X_test_embeddings['outputs'].numpy()

class TFNaiveBayesClassifier:
    dist = None

    # X is the matrix containing the vectors for each sentence
    # y is the list target values in the same order as the X matrix
    def fit(self, X, y):
        unique_y = np.unique(y)  # unique target values: 0,1
        print(unique_y)
        # `points_by_class` is a numpy array the size of
        # the number of unique targets.
        # in each item of the list is another list that contains the vector
        # of each sentence from the same target value
        points_by_class = np.asarray([np.asarray([np.asarray(X.iloc[x,:]) for x in range(0,len(y)) if y[x] == c]) for c in unique_y])
        mean_list=[]
        var_list=[]
        for i in range(0, len(points_by_class)):
            mean_var, var_var = tf.nn.moments(tf.constant(points_by_class[i]), axes=[0])
            mean_list.append(mean_var)
            var_list.append(var_var)
        mean=tf.stack(mean_list, 0)
        var=tf.stack(var_list, 0)
        # Create a 3x2 univariate normal distribution with the
        # known mean and variance
        self.dist = tfp.distributions.Normal(loc=mean, scale=tf.sqrt(var))

    def predict(self, X):
            assert self.dist is not None
            nb_classes, nb_features = map(int, self.dist.scale.shape)

            # uniform priors
            priors = np.log(np.array([1. / nb_classes] * nb_classes)).astype('float32')

            # Conditional probabilities log P(x|c)
            # (nb_samples, nb_classes, nb_features)
            all_log_probs = self.dist.log_prob(tf.reshape(tf.tile(X, [1, nb_classes]), [-1, nb_classes, nb_features]))
            # (nb_samples, nb_classes)
            cond_probs = tf.reduce_sum(all_log_probs, axis=2)

            # posterior log probability, log P(c) + log P(x|c)
            joint_likelihood = tf.add(priors, cond_probs)

            # normalize to get (log)-probabilities
            norm_factor = tf.reduce_logsumexp(joint_likelihood, axis=1, keepdims=True)
            log_prob = joint_likelihood - norm_factor
            # exp to get the actual probabilities
            return tf.exp(log_prob)



tf_nb = TFNaiveBayesClassifier()
tf_nb.fit(pd.DataFrame(x_train_matrix),y_train)
y_pred = tf_nb.predict(X_test_matrix)
predProbGivenText_df = pd.DataFrame(y_pred.numpy())

label_pred = []
for i in range(len(predProbGivenText_df[0])):
    if predProbGivenText_df[0][i]>predProbGivenText_df[1][i]:
        label_pred.append(0)
    elif predProbGivenText_df[0][i]<predProbGivenText_df[1][i]:
        label_pred.append(1)
    else:
        label_pred.append(0.5)

for i in range(len(post_author)):
    auth = post_author[i].split("\n")
    post_author[i] = auth[0]


test['Author'] = post_author
test['labels_NB'] = label_pred

for i in range(len(test[0])):
    if test['labels_NB'][i] == 1:
        print(test['Author'][i], "wrote: ",test[0][i])