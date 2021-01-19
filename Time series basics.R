
# R code Assignment 1

#############################################
# Part 1 

# Acquire the Covid-19 data
Covid19 <- read.csv("https://open-covid-19.github.io/data/data.csv")
summary(Covid19);nrow(Covid19)

# Select US data
X <- Covid19[Covid19$CountryCode=="US",]
X <- X[X$Key=="US",]
summary(X);nrow(X)

# Select Date, Confirmed, and Population
X <- X[,c("Date","Confirmed","Population")]
X <- na.omit(X)
X$Date <- as.Date(X$Date)
summary(X);nrow(X)

# Time series plot

require(ggplot2)    # must be installed before require will work
require(gridExtra)
grid.arrange(
  ggplot() + geom_point(data = X, aes(x = Date, y = Confirmed))+ ggtitle("Confirmed U.S COVID-19 Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot() + geom_point(data = X, aes(x = Date, y = Population)) + ggtitle("COVID-19 Data: US Population") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ncol=2,nrow=1)

# Any missing dates and does t+1 - t = c
head(X);tail(X)
dif <- diff(as.Date(X$Date,"%Y-%m-%d"))
dif
nrow(X)
table(dif)   # nrow - 1?


#############################################
# Part 2

# basic stats, histograms, and QQ plots of raw data
require(fBasics)

basicStats(X$Confirmed)
require(gridExtra)
grid.arrange(
  ggplot(X, aes(x=Confirmed)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6) + ggtitle("Histogram of Confirmed U.S COVID-19 Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X) + geom_qq(aes(sample = Confirmed)) + ggtitle("QQ Plot of Confirmed U.S COVID-19 Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ncol=2,nrow=1)

basicStats(X$Population)

grid.arrange(
  ggplot(X, aes(x= Population)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6)+ ggtitle("Histogram of U.S Population") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X) + geom_qq(aes(sample = Population)) + ggtitle("QQ Plot of U.S Population") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ncol=2,nrow=1)

t.test(X$Confirmed)
t.test(X$Population)

library(moments)
skewness(X)
kurtosis(X)

#############################################
# Part 3

y <- log(X$Confirmed+1)

basicStats(y)

grid.arrange(
  ggplot() + geom_point(data = X, aes(x = Date, y = y)) + ggtitle("Log Confirmed U.S Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X, aes(x= y)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6) + ggtitle("Histogram Log Confirmed U.S Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X) + geom_qq(aes(sample = y)) + ggtitle("QQ Plot Log Confirmed U.S Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ncol=3,nrow=1)

t.test(y)
######################

x1 <- log(X$Population+1)

basicStats(x1)

grid.arrange(
  ggplot() + geom_point(data = X, aes(x = Date, y = x1)) + ggtitle("Log U.S Population") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X, aes(x= x1)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6) + ggtitle("Histogram Log Population") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ggplot(X) + geom_qq(aes(sample = x1)) + ggtitle("QQ Plot Log Confirmed U.S Cases") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
  ncol=3,nrow=1)

t.test(x1)





#############################################
# Part 4

install.packages("fpp")
require(fpp)
require(forecast)

# https://www.ncdc.noaa.gov/cag/global/time-series
# Units: Degrees Celsius
# Base Period: 1901-2000
# Missing: -999

Climate <- read.csv("D:/Northwestern/MSDS 413/Week 1/ClimateData.csv",header=T)
X <- Climate
str(X)
# convert years into decades
X <- X[X$Year<2020,]
decade <- rep(1:trunc(nrow(X)/10),each=10)
X$Year <- decade/100 + X$Year
require(ggplot2)    # must be installed before require will work

### Plot the Data
ggplot() + geom_point(data = X, aes(x = Year, y = Value)) + 
	stat_smooth(aes(x = X$Year, y = X$Value), colour="red") +
	theme(axis.text.x = element_text(angle = 90, hjust = 1)) + 
  ggtitle("Average Annual Global Temperature by Year") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5))

# EDA here
basicStats(X$Value)
t.test(X$Value)
# histogram here
ggplot(X, aes(x= Value)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6)+ ggtitle("Histogram of Global Temperature") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5))
# normal Q-Q plot here
ggplot(X) + geom_qq(aes(sample = Value)) + ggtitle("QQ Plot Global Temperature") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5))

# make X a time series
X <- ts(X, frequency=10)  # cycles by decade
str(X)
main <- "Annual Temperature (C) vs. Decade"
autoplot(X[,2], main=main, ylab="Annual Temperature (C)", xlab="Decade")

# Make s.window as large as possible while keeping trend smooth
decomp <- stl(X[,2], s.window="periodic")
autoplot(decomp)
summary(decomp)

y <- decomp$time.series[,3]   # remainder

basicStats(y)

grid.arrange(
ggplot(as.data.frame(y), aes(x= y)) + geom_histogram(aes(y=..density..)) + geom_density(alpha=0.6) + ggtitle("Histogram of Global Temperature Residuals") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
ggplot() + geom_qq(aes(sample = y)) + ggtitle("QQ Plot of Global Temperature Residuals") +  theme(plot.title=element_text(size = 10,lineheight=0.6, face="bold", hjust=0.5)),
ncol = 2, nrow =1)

t.test(y)


