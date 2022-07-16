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

## NAMED ENTITY RECOGNITION (NER)

Named entity recognition is the task that implies identification entities in a sentence (like a person, organization, date, location, time, etc.), and their classification into categories.

Example:

![Screenshot 2022-07-17 at 4 27 36 AM](https://user-images.githubusercontent.com/25201417/179374328-5d877d8b-7651-429e-af3e-1f79520778eb.png)


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
[at last]
[include custom diagram , explain each step] 

##  Technologies Used

* Kafka
* Spark
* Python Libraries : tweepy, Flask, kafka, spacy, sklearn, keras, numpy, pyspark, nltk, matplotlib, os
 
## Kafka

## Tweepy API

## Spark Integration

## Implementation and Deployment of Pipeline

### Code Overview

### Running and Deployment of Application




