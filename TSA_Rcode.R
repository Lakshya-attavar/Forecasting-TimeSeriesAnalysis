#PART_1
#read the data
pcedata <- read.csv('PCE.csv')
colnames(pcedata)
#create a Time series object
pceTS <- ts(pcedata$PCE, start= c(1959,1), end = c(2023, 11), frequency=12)
pceTS
plot(pceTS)
#missing values
sum(is.na(pcedata$PCE))

#imputation
library(imputeTS)
pceComplete<-na_interpolation(pceTS)
plot(pceComplete)

#check for seasonality
library(forecast)
seasonplot(pceComplete) # no variation in data within each year
#decomposition
de <- decompose(pceComplete, type="additive")
plot(de)

#splitting train and test set as 80/20
nrow(pcedata)
size_train <- round(0.80 * nrow(pcedata))
size_train
train <-subset(pceComplete, end = size_train)
test<- pceComplete[(size_train+1):nrow(pcedata)]
size_test<-length(test)
size_test

#naive model
fcnaive <- naive(train, h = size_test)
fcnaive
plot(fcnaive)
summary(fcnaive)
accuracy(fcnaive,pceComplete)

#drift model
fcdrift <- rwf(train, h = size_test, drift = TRUE)
fcdrift
plot(fcdrift)
summary(fcdrift)
accuracy(fcdrift,pceComplete)

#exponential
fcholt <- holt(train, h= size_test)
summary(fcholt)
plot(fcholt)
accuracy(fcholt, pceComplete)

#ARIMA
tsdisplay(pceComplete)
afit <- auto.arima(train)
summary(afit)
checkresiduals(afit)
fcarima <-forecast(afit,h=size_test)
fcarima
accuracy(fcarima, pceComplete)
plot(fcarima)
autoplot(pceComplete) + autolayer(fcnaive$mean) + autolayer(fcholt$mean) + autolayer(fcarima$mean)
autoplot(pceComplete) + autolayer(fcdrift$mean) + autolayer(fcholt$mean) + autolayer(fcarima$mean)
#prediction using holt
holt_2024 <- holt(train,h = (size_test+11))
holt_2024
plot(holt_2024)
#accuracy(holt_2024)
#october 2024 = 16052.81

#One-Step ahead rolling forecast
#rolling forecatsing errors
e_naive <- tsCV(pceComplete, naive, h=1, window=size_train)
e_naive[1:779]

e_drift <- tsCV(pceComplete, rwf, drift = TRUE, h = 1, window = size_train)
e_drift[1:779]

e_holt <- tsCV(pceComplete, holt, h=1, window=size_train)
e_holt[1:779]

far2 <- function(x, h){forecast(auto.arima(x), h=h)}
e_arima <- tsCV(pceComplete, far2, h=1, window=size_train)
e_arima[1:779]

#train set for rolling forecast
train_rolling <- window(pceComplete, end = 2010.99)

#naive model
fit_naive <- naive(train_rolling)
refit_naive <- naive(pceComplete, model = fit_naive)
fc_rolling_naive <- window(fitted(refit_naive), start=2011)
fc_rolling_naive

#drift model
fit_drift <- rwf(train_rolling, drift = TRUE)
refit_drift <- rwf(pceComplete, model = fit_drift)
fc_rolling_drift <- window(fitted(refit_drift), start = 2011)
fc_rolling_drift

#holt method
fit_holt <- holt(train_rolling)
refit_holt <- holt(pceComplete, model = fit_holt)
fc_rolling_holt <- window(fitted(refit_holt), start=2011)
fc_rolling_holt 
#arima
fit_arima <- auto.arima(train_rolling)
refit_arima <- Arima(pceComplete, model=fit_arima)
fc_rolling_arima <- window(fitted(refit_arima), start=2011)
fc_rolling_arima

#accuracy
accuracy(fc_rolling_naive, pceComplete)
accuracy(fc_rolling_drift, pceComplete)
accuracy(fc_rolling_holt , pceComplete)
accuracy(fc_rolling_arima , pceComplete)

autoplot(pceComplete) + autolayer(fc_rolling_naive) + autolayer(fc_rolling_holt) + autolayer(fc_rolling_arima)
autoplot(pceComplete) + autolayer(fc_rolling_drift) + autolayer(fc_rolling_holt) + autolayer(fc_rolling_arima)


#PART_2

# Installing packages
install.packages('dplyr')
install.packages('stringr')
install.packages('RColorBrewer')
install.packages('topicmodels')
install.packages('ggplot2')
install.packages('LDAvis')
install.packages('servr')
install.packages('wordcloud')
install.packages('textcat')
install.packages('ldatuning')


#loading packages
library(dplyr) # basic data manipulation
library(tm) # package for text mining package
library(stringr) # package for dealing with strings
library(RColorBrewer)# package to get special theme color
library(topicmodels) # package for topic modelling
library(ggplot2) # basic data visualization
library(LDAvis) # LDA specific visualization 
library(servr) # interactive support for LDA visualization
library(textcat) #separate English review
library(ldatuning)
library(wordcloud) # package to create wordcloud

#set seed for reproducibility
set.seed(205)

#read the data
hotel_data <- read.csv(file = "HotelsData.csv", header = TRUE)
nrow(hotel_data)
#detect languages in the hotel reviews
detect_languages <-textcat(hotel_data$Text.1)
#detect_languages
hotel_data$language <-detect_languages
colnames(hotel_data)
#retaining only english reviews
data_english <- hotel_data[hotel_data$language == "english",]
colnames(data_english)
test <- sample_n(data_english, 2000)

#split the test dataset of 2000 records into positive and negative reviews
#considering review score 3, 4, 5 as positive and 1,2 as negative
positive_reviews <- test[test$Review.score >= 3, ]
negative_reviews <- test[test$Review.score <= 2,]

#function to create document term matrix, wordcloud and find number of topics
topic_number_fn <- function(reviews) {
  document <- stringr::str_conv(reviews$Text.1, "UTF.8")
  doc_corpus <-Corpus(VectorSource(document))
  dtm_reviews <- DocumentTermMatrix(doc_corpus,
                                    control = list(lemma=TRUE,removePunctuation = TRUE,
                                                   removeNumbers = TRUE, stopwords = TRUE,
                                                   tolower = TRUE))
  
  row.sum=apply(dtm_reviews , 1, FUN=sum)
  dtm_docs = dtm_reviews[row.sum != 0,]
  
  
  dtm_new <- as.matrix(dtm_docs)
  frequency <-colSums(dtm_new)
  frequency <-sort(frequency, decreasing = TRUE)
  document_length <- rowSums(dtm_new)
  frequency[1:10]
  #word cloud
  words <- names(frequency)
  wordcloud(words[1:50], frequency[1:50], rot.per = 0.15, 
            random.order = FALSE, scale = c(4, 0, 5), 
            random.color = FALSE, colors= brewer.pal(8, "Dark2"))
  #finding topic number
  topic_num <- FindTopicsNumber(
    dtm_new,
    topics = seq(from = 5, to = 25, by = 1),
    metrics = c("Griffiths2004", "CaoJuan2009", "Arun2010"),
    method = "Gibbs",
    control = list(seed = 205),
    mc.cores = 2L,
    verbose = TRUE
  )
  
  #plot of topic numbers
  FindTopicsNumber_plot(topic_num)
  return (list(dtm_docs, document_length, frequency))
}

positive_topics_dtm <- topic_number_fn(positive_reviews)
positive_topics_dtm[[3]][1:10]
negative_topics_dtm <- topic_number_fn(negative_reviews)
negative_topics_dtm[[3]][1:10]

#Topic modelling function
function_topics <- function(topic_dtm, k, reviews){
  lda_model <-LDA(topic_dtm[[1]], k, method="Gibbs", 
                  control=list(iter=1000,seed=1000))
  phi <- posterior(lda_model)$terms %>% as.matrix 
  theta <- posterior(lda_model)$topics %>% as.matrix 
  ldamodel_terms <- as.matrix(terms(lda_model, 10))
  
  # Which 'topic' is the review in (highest probability)-> To get grouping of reviews by topics
  lda_model.topics <- data.frame(topics(lda_model))
  lda_model.topics$index <- as.numeric(row.names(lda_model.topics))
  reviews$index <- as.numeric(row.names(reviews))
  data_with_topic <- merge(reviews, lda_model.topics, by='index',all.x=TRUE)
  data_with_topic <- data_with_topic[order(data_with_topic$index), ]
  data_with_topic[0:10,]
  
  # association of reviews with topics
  topic_Probabilities <- as.data.frame(lda_model@gamma)
  
  topic_Probabilities_prev <- (colMeans(lda_model@gamma))
  ordered_topics <- order(topic_Probabilities_prev, decreasing = TRUE)
  ordered_topics 
  Top_ordered_topics <-for(i in 1:length(ordered_topics)){
    cat ("Topic", ordered_topics[i],":",topic_Probabilities_prev[ordered_topics[i]], "\n")
  }
  
  vocab <- colnames(phi) #vocab list in DTM
  
  # create the JSON object to feed the visualization in LDAvis:
  json_ldamodel <- createJSON(phi = phi, theta = theta, 
                              vocab = vocab, doc.length = topic_dtm[[2]], 
                              term.frequency = topic_dtm[[3]])
  
  
  serVis(json_ldamodel, out.dir = 'vis', open.browser = TRUE)
  return(list(ldamodel_terms,data_with_topic,topic_Probabilities ))
}

#topic modelling for positive terms
positive_terms<-function_topics(positive_topics_dtm, 20, positive_reviews)
positive_terms[1]
positive_terms[[2]][0:10,]
positive_terms[[3]][0:10,1:20]

negative_terms<-function_topics(negative_topics_dtm, 15, negative_reviews)
negative_terms[1]
negative_terms[[2]][0:10,]
negative_terms[[3]][0:10,1:15]
