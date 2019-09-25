#Huang Xueyan
library(tidyverse)
library(reshape2)
library(sentimentr)
getwd()
setwd("C:/Users/spfbda/Desktop/")


tweet=read.csv("stockerbot-export.csv") %>% mutate(text=paste(text))

#sentiment(tweet$text[4])
#table(tweet$timestamp)

#sentiment analysis for all stocks:

pickN=1000;
tweet_small=tweet[1:pickN,] %>% filter(symbols!="",company_names !="")

#out=sentiment_by(text.var = tweet_small$text,by=tweet_small$company_names)
out=sentiment_by(text.var = tweet_small$text%>%get_sentences, by=tweet_small$company_names)