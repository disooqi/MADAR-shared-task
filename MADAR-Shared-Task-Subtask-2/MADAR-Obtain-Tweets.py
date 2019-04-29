# -*- coding: utf-8 -*-

'''Use of this script will require a Twitter developer's account
and corresponding authentication credentials. The authentication 
credentials have to be provided in lines 81-84 of the code. More 
information about creating Twitter developer's account and obtaining 
authentication credentials can be found in the README.txt file'''


import unicodecsv as csv
import json
import time
import requests
import sys
import tweepy
from tweepy import OAuthHandler


def print_usage():
    ''' prints instructions for using the script'''

    print (
    '''Usage:
    python3 MADAR-Obtain-Tweets.py  <input_file> <output_file>
    
    The input file must be in the same format as the
    MADAR-Twitter-Subtask-2.TRAIN.user-tweet-features.tsv
    or MADAR-Twitter-Subtask-2.DEV.user-label.tsv.

    The output file is the name of the file where the obtained tweets
    will be appended. If any tweet is unavailable, it will write
    <UNAVAILABLE> for that tweet.

        ''')


def get_tweet_text(tweets, tweet_id):
    ''' finds tweet corresponding to the tweet_id in tweets and
        returns the text of the tweet. if the tweet is not available,
        returns <UNAVAILABLE> '''
    
    for tweet in tweets:
        if tweet == None:
            continue
        
        if (tweet.id_str == tweet_id):
            text = tweet.text
            if text == None:
                return "<UNAVAILABLE>"
            
            return text
    return "<UNAVAILABLE>"


def write_batch(rows, tweets, csvwriter):
    ''' writes batch of tweets alongside their information to file'''

    for row in rows:
        tweet_id = row[1]
        tweet_text = get_tweet_text(tweets, tweet_id)
        tweet_text = tweet_text.replace('\n', ' ').replace('\r', '')
        current_row = row + [tweet_text]
        csvwriter.writerow(current_row)
    

def get_batch_tweets(api, rows):
    ''' gets tweets in a batch from Twitter using tweet ids'''
    
    tweet_ids = []
    for row in rows:
        tweet_ids.append(row[1])

    tweets = api.statuses_lookup(tweet_ids)
    return (tweets)
    
    

if __name__ == '__main__':

    ## Authentication details
    consumer_key = ''
    consumer_secret = ''
    access_token = ''
    access_secret = ''

    if (len (sys.argv) != 3):
            print_usage()
            exit()
    
    readfile = sys.argv[1]
    writefile = sys.argv[2]

    ## Authentication
    auth = OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_secret)
    api = tweepy.API(auth, wait_on_rate_limit = True, wait_on_rate_limit_notify = True)

    ## obtain tweets from Twitter
    i = 0
    batch_count = 0
    batch_rows = []
    with open(readfile, "rb") as csvread:
        with open(writefile, "wb") as csvwrite:
            csvreader = csv.reader(csvread, delimiter="\t", encoding = "utf-8")
            csvwriter = csv.writer(csvwrite, delimiter="\t", encoding = "utf-8")

            for row in csvreader:
                
                ## writes header
                if (i == 0):
                    num_cols = len(row)
                    current_row = row + ["#"+ str(num_cols + 1)+ " Tweet Text"]
                    csvwriter.writerow(current_row)
                    i += 1
                else:
                    ## writes batches of 100 tweets and their information
                    batch_rows.append(row)
                    batch_count += 1
                    if (batch_count == 100):
                        tweets = get_batch_tweets(api, batch_rows)
                        write_batch(batch_rows, tweets, csvwriter)
                        batch_count = 0
                        batch_rows = []
            
            ## writes the last batch 
            if (batch_count > 0):
                tweets = get_batch_tweets(api, batch_rows)
                write_batch(batch_rows, tweets, csvwriter)
