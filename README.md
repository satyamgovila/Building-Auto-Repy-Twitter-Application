# Building-Auto-Reply-Twitter-Application

## Buisness Use Case

Natural Language Processing (NLP) is a sub part of Artificial Intelligence that uses algorithms to understand and process human language. Various computational methods are used to process and analyze human language and a wide variety of real-life problems are solved using Natural Language Processing.
For businesses that deal with a large amount of unstructured data, in the forms of emails, social media, chats, surveys, etc. Natural Language Processing tools are an asset. Workplaces can gain valuable insights that would help derive crucial business decisions and automate tasks.

In this project, we aim to explore an important buisness use case that leverage the power of Sentiment Analysis. The project focusses on designing and developing a Deep Learning based application that automatically replies to query-related tweets with a trackable ticket ID generated based on the query category. 

For example, 
During covid times, many people were reporting complains and issues regarding fligts, prices, seat availability , reschedule and refund, etc. due to which it became very tedious and time taking for Airline organisations to read every tweet and reply with a ticket number. 

These queries are product specific and our application automatically replies to these queries by analysing the sentiment of the query and assigning a ticket number, which in turn saves a lot of time for the organisation which is wasted in replying to the tweets and assigning a ticket number for it.

In this project, we take Airlines Dataset as our use case and develop a deep learning application that is developed on an end-to-end pipeline on cloud platform.

## Objective

To design and develop a deep learning application that automatically replies to query-related tweets with a trackable ticket ID generated based on the query category predicted using deep learning models. The whole pipeline is automated with tweepy API and Big Data techniques with final deployment on AWS.

## Sentiment Analysis 

Sentiment analysis--also known as conversation mining-- is a technique that lets you analyze ​​opinions, sentiments, and perceptions. In a business context, Sentiment analysis enables organizations to understand their customers better, earn more revenue, and improve their products and services based on customer feedback.

In this project, we are going to use one of the most common yet popular standard technique of Sentiment Analysis where on given a text based tweet as input, the process recognizes the overall tone of a written text and classifies it as positive, negative, or neutral.

Sentiment analysis (SA) works by using Machine learning and its constituent Deep learning algorithms to create SA models. These models are trained by feeding it millions of pieces of text to detect if a message is positive, negative, or neutral. Sentiment analysis works by breaking a message down into topic chunks and then assigning a sentiment score to each topic.

Using SA is very beneficial for any brand, if the organisation uses SA to track negative sentiment towards their brand. This not only lets them fix their product/service but also enables them to understand how customer sentiment towards their brand changes over time.


## Data Overview 

**Data Source** : https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment

For the training purposes in this project, we have used an “airline tweets” dataset, which includes necessary fields like airline names as tags, actual text, sentiment class, i.e., positive, negative, or neutral and a topic class which can be one of these:

1. Baggage Issue
2. Customer Experience 
3. Delay and Customer Service
4. Extra Charges
5. Online Booking
6. Reschedule and Refund
7. Reservation Issue
8. Seating Preferences

## System Architecture 

[include custom diagram , explain each step] 

In the system architecture , we are using Kafka Topic to ingest any new tweet using Kafka Topic with a custom class named Tweet Listener which reads data as soon as there is any new event i.e. new tweet and this Tweet Listener is stored in Kafka Topic. This Kafka Topic is further streamed using Spark Layer.

This Spark Layer has many components such as:
* Spark Streaming : used to extraxt data from Kafka Topic in JSON format 

* Keras Model Sentiment Analysis : Receives data from Spark Streaming and passes through user defined functions in this model

Reference to ipynb Notebook :  notebook/BinaryCLassificationKeras.ipynb (dataset : dataset/Tweets.csv)
This notebook demonstrates data pre-processing and evaluating binary inference (whether tweet is negative/positive ) using the following steps :
  1. Reading the dataset
  2. Removing all "neutral" tweets from df, from "airline_sentiment" column
  3. Preprocessing the df using nltk, padding the tweets , and data cleaning using regex
  4. Building and compiling LSTM Model with "softmax" activation and loss function as "binary_crossentropy"
  5. Predicting the sample complaint and classifying as "positive" or "negative" tweet.
  6. Model will be saved in binaryClassificationModel.h5 file and tokenizer value in tokenizerBinaryClassification.pickle file
  Note : Here, in model.add(Dense(2,activation='softmax')) , we have used 2 because it is binary classification , if we had considered neutral also then it will be 3
  


* Keras Complain Classification : used to label the parsed tweet to get the category of the compain from pre defined list of categories, for example, complain can be related to refunds, airline departure, etc.

* LDA Data Labelling :
Reference to notebook : ntoebook/LDA_Model_building.ipynb
Output Dataset from LDA model : dataset/labelled_airline_tweet.csv

**Topic Labeling** :- It is a NLP technique that uses unsupervised learning to extract text and classify it into topics. It is a pretty useful technique when dealing with large volumes of unstructured text data where sieving out data manually is time-consuming and labor-intensive. 

**Pre-Processing** :- The approach of pre-processing is using Nltk stopwords to remove the common helping verbs, as well as pass-through Gensim simple preprocess tools to discard any tokens shorter than the minimum length of X characters as well as punctuations. 

Latent Dirichlet Allocation aka LDA is a type of topic modeling algorithm employed to identify topics within documents.

By setting num_topics, we are deciding on a preset number of topics to determine. Next, LDA will randomly assign documents to a topic, then calculate the probability of a word being in a topic and probability of topic occurrence given a specific document. 

This process is iterated based on the number of iterations which will improve the performance of LDA.

```
# Build LDA model
lda_model4 = gensim.models.ldamodel.LdaModel(corpus=corpus,
                                           id2word=id2word,
                                           num_topics=8, 
                                           random_state=123,
                                           update_every=1,
                                           chunksize=10,
                                           passes=10,
                                           alpha='auto',
                                           eta='auto',
                                           iterations=125,
                                           per_word_topics=True)
```


* Entitiy Extraction : This component is used to extract information about the entity in the tweet such as source and destination place of the user , duration of flight, kind of issue faced , etc

* API Call and Response : This API is built and triggered automatically to generate ticket number and POST it automatically to tweet as a reply to user.

* O/P Data Storage :  This is used to store the final enriched output data in parquet format from complete spark job lifecycle which can be finally analysed by the organisation. The organisation can get a highly enriched data from this DB which will have information such as ticket number, problem type, name of person, etc.

## NAMED ENTITY RECOGNITION (NER)

Named entity recognition is the task that implies identification entities in a sentence (like a person, organization, date, location, time, etc.), and their classification into categories.

Example:

![Screenshot 2022-07-17 at 4 27 36 AM](https://user-images.githubusercontent.com/25201417/179374328-5d877d8b-7651-429e-af3e-1f79520778eb.png)

For this project, we are going to perform NER process using Spacy. Spacy is an open-source Natural Language Processing library that can be used for various tasks. It has built-in methods for Named Entity Recognition. Spacy has a fast statistical entity recognition system.

Note :-  
Statistical Models in spaCy used in this project "en_core_web_sm"  (English multi-task CNN trained on OntoNotes. Size – 11 MB)


##  Technologies Used

* Kafka
* Spark
* Python Libraries : tweepy, Flask, kafka, spacy, sklearn, keras, numpy, pyspark, nltk, matplotlib, os

## Tweepy API

Tweepy is an open source Python package that gives us a very convenient way to access the Twitter API with Python. It includes a set of classes and methods that represent Twitter's models and API endpoints, and it transparently handles various implementation details, such as: Data encoding and decoding , OAuth authentication , HTTP requests, etc.

```
! pip install tweepy

import tweepy

# Variables that contains the credentials to access Twitter API
ACCESS_TOKEN = 'your_access_token'
ACCESS_SECRET = 'your_access_secret'
CONSUMER_KEY = 'your_consumer_key'
CONSUMER_SECRET = 'your_consumer_secret'

# Setup access to API
def connect_to_twitter_OAuth():
    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_SECRET)

    api = tweepy.API(auth)
    return api
    
# Create API object
api = connect_to_twitter_OAuth()

```
 
## Kafka
(Setting up and Running Reference : https://kafka-python.readthedocs.io/en/master/ )

Apache Kafka is an open-source stream-processing software platform which aims to provide a unified, high-throughput, low-latency platform for handling real-time data feeds. Its storage layer is essentially a “massively scalable pub/sub message queue architected as a distributed transaction log” making it highly valuable for enterprise infrastructures to process streaming data. Additionally, Kafka connects to external systems (for data import/export) via Kafka Connect and provides Kafka Streams, a Java stream processing library.

### Key Terminologies in Kafka

* Topics : It is basically a stream of records, and every message that is feed into the system must be part of some topic.The messages are stored in key-value format with a assigned sequence value, Offset. 
* Producers : Producers are the apps responsible to publish data into Kafka system.
* Consumers : Messages published into topics are then utilized by Consumers apps.
* Broker : Every instance of Kafka that is responsible for message exchange is called a Broker. 

```
!pip install kafka-python

from kafka import KafkaProducer

producer = KafkaProducer(  
    bootstrap_servers = ['localhost:9092'], # setting the host & port to contact the producer to bootstrap initial cluster.Default host & port is localhost: 9092.
    value_serializer = lambda x:dumps(x).encode('utf-8')  # transform the data into a JSON file and encode it to UTF-8
    )  

```

## Spark Integration

Refer to script : source/Engine_SparkJob.py

**Apache Spark** is a lightning-fast cluster computing designed for fast computation. It was built on top of Hadoop MapReduce and it extends the MapReduce model to efficiently use more types of computations which includes Interactive Queries and Stream Processing.

**Spark Streaming** is an extension of the core Spark API that provides scalable, high-throughput and fault-tolerant stream processing of live data streams. Data ingestion can be done from many sources like Kafka, Apache Flume, Amazon Kinesis or TCP sockets and processing can be done using complex algorithms that are expressed with high-level functions like map, reduce, join and window. 

Spark Straming helps in Dynamic load balancing , Fast failure and straggler recovery , Advanced analytics like machine learning and interactive SQL , etc.
In Spark Streaming divide the data stream into batches called DStreams, which internally is a sequence of RDDs. The RDDs process using Spark APIs, and the results return in batches.



![1_FLYjc6U-qAQ64yDLLrzdWw](https://user-images.githubusercontent.com/25201417/179388660-3f2c0e85-8c15-43ad-8260-1584af71fc23.jpeg)



**Spark Submit Job**

In order to run the python script on Spark and submit the spark job, we need to execute the following commads : 

>> ./lib/python3.7/site-packages/pyspark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2  ./code/[python_file].py localhost 9999 --master local[*]

On installing **pyspark** package, we need to specify the name and location of the **spark-submit** jar ,and then we need to specify the package which is in the given format : --packages [organisation name]:[spark package]:[version]
Here, we also specify the localhost and master as local[ * ] which means allocating all the ports from the local.

>> ./lib/python3.7/site-packages/pyspark/bin/spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 --py-files ./modular/source/MLPipeline.zip  ./modular/source/Engine_SparkJob.py localhost 9999 --master local[*]

# spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.1.2 --py-files MLPipeline.zip  Engine_SparkJob.py localhost 9999

# kafka-console-consumer.sh --topic tweet-data --bootstrap-server localhost:9092



### Code Overview

### Running and Deployment of Application

![image](https://user-images.githubusercontent.com/25201417/179390210-ccf0d2e7-74a5-4481-a1b2-4c6283694db4.png)

In order to run and deploy the application , we will go through the following steps :- 

1. Setup two AWS EC2 instances, on first instance , we will create the Kafka instance and HOST API call script, Tweet Listener class  will be deployed on this instance. The setup will be configured on the basis of choosing Instance Type, storage , security group 

Execute the following steps on the first instance :-  
```
sudo apt update
sudo apt install default-jdk 
sudo wget https://packages.confluent.io/archive/5.5/confluent-5.5.0-2.12.tar.gz
sudo tar xvzf confluent-5.5.0-2.12.tar.gz \
sudo vi /etc/profile
export PATH=/home/ubuntu/confluent-5.5.0/bin:$PATH
source /etc/profile
echo $PATH
/home/ubuntu/confluent-5.5.0/bin/confluent-hub install --no-prompt confluentinc/kafka-connect-datagen:0.1.0
/home/ubuntu/kafka/kafka/confluent-5.5.0/bin/confluent start
confluent local start

```

# kafka-console-consumer.sh --topic tweet-data --bootstrap-server localhost:9092


2. Spark AWS Submit
 




