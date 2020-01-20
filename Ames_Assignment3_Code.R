
setwd("D:/Northwestern/WD")

##################

mydata <- read.csv(file="D:/Northwestern/MSDS 410/Week 1/Assignment #1/ames_housing_data.csv",head=TRUE,sep=",")

str(mydata)
head(mydata)
names(mydata)
mydata$TotalFloorSF <- mydata$FirstFlrSF + mydata$SecondFlrSF
mydata$HouseAge <- mydata$YrSold - mydata$YearBuilt
mydata$QualityIndex <- mydata$OverallQual * mydata$OverallCond
mydata$logSalePrice <- log(mydata$SalePrice)
mydata$price_sqft <- mydata$SalePrice/mydata$TotalFloorSF
mydata$Bathrooms <- mydata$FullBath + 0.5*mydata$HalfBath
summary(mydata$price_sqft)

### FILTER CONDITIONS###

test0 <- mydata[mydata$price_sqft >= 50,]
test0 <- mydata[mydata$price_sqft <= 250,]

Filter0 <- mydata[mydata$TotalFloorSF <= 4000,]
Filter1 <- Filter0[Filter0$TotalFloorSF > 500,]
Filter2 <- Filter1[Filter1$SalePrice <= 500000,]
Filter3 <- Filter2[Filter2$SalePrice > 50000,]
Filter4 <- Filter3[Filter3$LotArea <= 25000,]
Filter5 <- Filter4[Filter4$LotArea > 1500,]
#Filter6 <- Filter5[Filter5$BldgType == "1Fam",]
Filter7 <- Filter5[Filter5$Bathrooms <4,]
Filter8 <- Filter7[Filter7$Bathrooms >=1,]
Filter9 <- Filter8[Filter8$price_sqft <= 250,]
Filter10 <- Filter9[Filter9$price_sqft >=50,]
#Filter11 <- Filter10[Filter10$Functional == "Typ",]
#Filter12 <- Filter11[Filter11$Zoning %in% c('RH', 'RL', 'RP', 'RM'),]
Filter13 <- Filter10[Filter10$Utilities == "AllPub",]


### EDA for Categorical variables (700x300 charts)

require(ggplot2)

ggplot(Filter13, aes(x=Neighborhood, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Neighborhood") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=BldgType, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by BldgType") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=LotShape, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by LotShape") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=HouseStyle, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by HouseStyle") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=RoofStyle, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by RoofStyle") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=SaleType, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by SaleType") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=Street, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Street") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=Foundation, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Foundation") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=BsmtQual, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by BsmtQual") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=Heating, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Heating") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=CentralAir, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by CentralAir") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=Electrical, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Electrical") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=KitchenQual, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by KitchenQual") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=GarageType, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by GarageType") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=PoolQC, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by PoolQC") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=SaleCondition, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by SaleCondition") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=Zoning, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Zoning") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

ggplot(Filter13, aes(x=LotConfig, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by LotConfig") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

###Categorical variables with potential

#BldgType
summary(Filter13$BldgType)

Filter13$d1_BldgType <- ifelse(Filter13$BldgType == "1Fam", 1, 0) #TwnhsE is base
Filter13$d2_BldgType <- ifelse(Filter13$BldgType == "2fmCon", 1, 0)
Filter13$d3_BldgType <- ifelse(Filter13$BldgType == "Duplex", 1, 0)
Filter13$d4_BldgType <- ifelse(Filter13$BldgType == "Twnhs", 1, 0)

library(plyr)
ddply(Filter13, .(BldgType), summarize,  Mean.SalePrice=mean(SalePrice), Mdeian.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.BldgType <- lm(data=Filter13, SalePrice ~ d1_BldgType + d2_BldgType + d3_BldgType + d4_BldgType)
summary(cat.lm.BldgType)
par(mfrow=c(2,2))
plot(cat.lm.BldgType)

#HouseStyle
summary(Filter13$HouseStyle)

Filter13$d1_HouseStyle <- ifelse(Filter13$HouseStyle == "1Story",1,0) #Other is base
Filter13$d2_HouseStyle <- ifelse(Filter13$HouseStyle == "2Story",1,0) 


library(plyr)
ddply(Filter13, .(HouseStyle), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.HouseStyle <- lm(data=Filter13, SalePrice ~ d1_HouseStyle + d2_HouseStyle)
summary(cat.lm.HouseStyle)
par(mfrow=c(2,2))
plot(cat.lm.HouseStyle)


#Central Air
summary(Filter13$CentralAir)

Filter13$d1_CentralAir <- ifelse(Filter13$CentralAir == "Y",1,0) #No is base

ddply(Filter13, .(CentralAir), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.CentralAir <- lm(data=Filter13, SalePrice ~ d1_CentralAir)
summary(cat.lm.CentralAir)
par(mfrow=c(2,2))
plot(cat.lm.CentralAir)

#KitchenQual

summary(Filter13$KitchenQual)

Filter13$d1_KitchenQual <- ifelse(Filter13$KitchenQual == "Ex",1,0) #Other is base (Fa & Po)
Filter13$d2_KitchenQual <- ifelse(Filter13$KitchenQual == "Gd",1,0) 
Filter13$d3_KitchenQual <- ifelse(Filter13$KitchenQual == "TA",1,0) 

ddply(Filter13, .(KitchenQual), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.KitchenQual <- lm(data=Filter13, SalePrice ~ d1_KitchenQual + d2_KitchenQual + d3_KitchenQual)
summary(cat.lm.KitchenQual)
anova(cat.lm.KitchenQual)
par(mfrow=c(2,2))
plot(cat.lm.KitchenQual)

#Lotshape

summary(Filter13$LotShape)

Filter13$d1_LotShape <- ifelse(Filter13$LotShape == "Reg",1,0) #Other is base (IR1, IR2, IR3)

ddply(Filter13, .(LotShape), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.LotShape <- lm(data=Filter13, SalePrice ~ d1_LotShape)
summary(cat.lm.LotShape)
par(mfrow=c(2,2))
plot(cat.lm.LotShape)

#Zoning
summary(Filter13$Zoning)

Filter13$d1_Zoning <- ifelse(Filter13$Zoning == "C (all)",1,0)  #RL is base
Filter13$d2_Zoning <- ifelse(Filter13$Zoning == "FV",1,0) 
Filter13$d3_Zoning <- ifelse(Filter13$Zoning == "RH",1,0) 
Filter13$d4_Zoning <- ifelse(Filter13$Zoning == "RM",1,0) 

ddply(Filter13, .(Zoning), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

cat.lm.Zoning <- lm(data=Filter13, SalePrice ~ d1_Zoning + d2_Zoning + d3_Zoning + d4_Zoning)
summary(cat.lm.Zoning)
anova(cat.lm.Zoning)
par(mfrow=c(2,2))
plot(cat.lm.Zoning)

###Unused categorical variables (tested for promise)
#Pool
summary(Filter13$PoolArea)

Filter13$d1_Pool <- ifelse(Filter13$PoolArea > 0 ,1,0)  #no pool is base not enough people with pools

#neighbourhood

Neighborhood.average <- aggregate(Filter13$SalePrice, list(Filter13$Neighborhood), mean)
summary(Neighborhood.average$x)

Neighborhood.average$class <- ifelse(Neighborhood.average$x <= 136965, "lower", ifelse(Neighborhood.average$x <= 185399, "lowermiddle", ifelse(Neighborhood.average$x <= 212965, "uppermiddle","upper")))
table(Neighborhood.average$class)

Filter13$Neighborhood.class <- with(Neighborhood.average, class[match(Filter13$Neighborhood, Group.1)])

summary(Filter13$Neighborhood.class)
table(Filter13$Neighborhood.class)

Filter13$d1_Neighborhood.class <- ifelse(Filter13$Neighborhood.class == "lowermiddle",1,0)  #lower is base
Filter13$d2_Neighborhood.class <- ifelse(Filter13$Neighborhood.class == "uppermiddle",1,0) 
Filter13$d3_Neighborhood.class <- ifelse(Filter13$Neighborhood.class == "upper",1,0) 


cat.lm.class <- lm(data=Filter13, SalePrice ~ d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class)
summary(cat.lm.class)
anova(cat.lm.class)
par(mfrow=c(2,2))
plot(cat.lm.class)

#######################################promise 

###Check interaction
#interaction between GrLivArea and Neighbourhood.Class

Train.clean$i1 <- Train.clean$GrLivArea * Train.clean$d1_Neighborhood.class
Train.clean$i2 <- Train.clean$GrLivArea * Train.clean$d2_Neighborhood.class
Train.clean$i3 <- Train.clean$GrLivArea * Train.clean$d3_Neighborhood.class

Train.clean$i4 <- Train.clean$GrLivArea * Train.clean$d1_KitchenQual
Train.clean$i5 <- Train.clean$GrLivArea * Train.clean$d2_KitchenQual
Train.clean$i6 <- Train.clean$GrLivArea * Train.clean$d3_KitchenQual

Train.clean$i7 <- Train.clean$GrLivArea * Train.clean$d1_Zoning
Train.clean$i8 <- Train.clean$GrLivArea * Train.clean$d2_Zoning

Train.clean$i9 <- Train.clean$HouseAge * Train.clean$d1_Neighborhood.class
Train.clean$i10 <- Train.clean$HouseAge * Train.clean$d2_Neighborhood.class
Train.clean$i11 <- Train.clean$HouseAge * Train.clean$d3_Neighborhood.class

Train.clean$i12 <- Train.clean$HouseAge * Train.clean$d1_KitchenQual
Train.clean$i13 <- Train.clean$HouseAge * Train.clean$d2_KitchenQual
Train.clean$i14 <- Train.clean$HouseAge * Train.clean$d3_KitchenQual

Train.clean$i15 <- Train.clean$HouseAge * Train.clean$d1_Zoning
Train.clean$i16 <- Train.clean$HouseAge * Train.clean$d2_Zoning

##################################

Filter13$i1 <- Filter13$GrLivArea * Filter13$d1_Neighborhood.class
Filter13$i2 <- Filter13$GrLivArea * Filter13$d2_Neighborhood.class
Filter13$i3 <- Filter13$GrLivArea * Filter13$d3_Neighborhood.class

Filter13$i4 <- Filter13$GrLivArea * Filter13$d1_KitchenQual
Filter13$i5 <- Filter13$GrLivArea * Filter13$d2_KitchenQual
Filter13$i6 <- Filter13$GrLivArea * Filter13$d3_KitchenQual

Filter13$i7 <- Filter13$GrLivArea * Filter13$d1_Zoning
Filter13$i8 <- Filter13$GrLivArea * Filter13$d2_Zoning

Filter13$i9 <- Filter13$HouseAge * Filter13$d1_Neighborhood.class
Filter13$i10 <- Filter13$HouseAge * Filter13$d2_Neighborhood.class
Filter13$i11 <- Filter13$HouseAge * Filter13$d3_Neighborhood.class

Filter13$i12 <- Filter13$HouseAge * Filter13$d1_KitchenQual
Filter13$i13 <- Filter13$HouseAge * Filter13$d2_KitchenQual
Filter13$i14 <- Filter13$HouseAge * Filter13$d3_KitchenQual

Filter13$i15 <- Filter13$HouseAge * Filter13$d1_Zoning
Filter13$i16 <- Filter13$HouseAge * Filter13$d2_Zoning

################################
interaction.lm1 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_KitchenQual + d2_KitchenQual + d3_KitchenQual)
summary(interaction.lm1)
anova(interaction.lm1)
par(mfrow=c(2,2))
plot(interaction.lm1)

interaction.lm2 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_KitchenQual + d2_KitchenQual + d3_KitchenQual +i1 + i2 + i3)
summary(interaction.lm2)
anova(interaction.lm2)
par(mfrow=c(2,2))
plot(interaction.lm2)

anova(interaction.lm1,interaction.lm2)
##########################################

interaction.lm3 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class)
summary(interaction.lm3)
anova(interaction.lm3)
par(mfrow=c(2,2))
plot(interaction.lm3)

interaction.lm4 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class +i4 + i5 + i6)
summary(interaction.lm4)
anova(interaction.lm4)
par(mfrow=c(2,2))
plot(interaction.lm4)

anova(interaction.lm3,interaction.lm4)

######################################################

interaction.lm5 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_Zoning + d2_Zoning)
summary(interaction.lm5)
anova(interaction.lm5)
par(mfrow=c(2,2))
plot(interaction.lm5)

interaction.lm6 <- lm(data = Train.clean, SalePrice ~ GrLivArea + d1_Zoning + d2_Zoning + i7 + i8)
summary(interaction.lm6)
anova(interaction.lm6)
par(mfrow=c(2,2))
plot(interaction.lm6)

anova(interaction.lm5,interaction.lm6)

################################
interaction.lm7 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_KitchenQual + d2_KitchenQual + d3_KitchenQual)
summary(interaction.lm7)
anova(interaction.lm7)
par(mfrow=c(2,2))
plot(interaction.lm7)

interaction.lm8 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_KitchenQual + d2_KitchenQual + d3_KitchenQual +i9 + i10 + i11)
summary(interaction.lm2)
anova(interaction.lm2)
par(mfrow=c(2,2))
plot(interaction.lm8)

anova(interaction.lm7,interaction.lm8)
##########################################

interaction.lm9 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class)
summary(interaction.lm9)
anova(interaction.lm9)
par(mfrow=c(2,2))
plot(interaction.lm9)

interaction.lm10 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class +i12 + i13 + i14)
summary(interaction.lm10)
anova(interaction.lm10)
par(mfrow=c(2,2))
plot(interaction.lm10)

anova(interaction.lm9,interaction.lm10)

######################################################
##no interaction
interaction.lm11 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_Zoning + d2_Zoning)
summary(interaction.lm11)
anova(interaction.lm11)
par(mfrow=c(2,2))
plot(interaction.lm11)

interaction.lm12 <- lm(data = Train.clean, SalePrice ~ HouseAge + d1_Zoning + d2_Zoning + i15 + i16)
summary(interaction.lm12)
anova(interaction.lm12)
par(mfrow=c(2,2))
plot(interaction.lm12)

anova(interaction.lm11,interaction.lm12)

########################################
correlations <- Train.clean[,c(2,4,8,9)]
require(corrplot)
mcor <- cor(correlations)
corrplot(mcor, method="shade", shade.col=NA, tl.col="black",tl.cex=1,number.cex = 1, addCoef.col = "black")

final.model <- lm(data=Train.clean, logSalePrice~ GrLivArea + HouseAge + TotalBsmtSF + QualityIndex + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class+  i1 +i2 +i3 +i9+i10+i11 )
summary(final.model)
par(mfrow=c(2,2))
plot(final.model)
accuracy(final.model)

influential <- list(ols_plot_dffits(final.model)$outliers)
influential <- data.frame(influential)

remove.dffits <- Train.clean[-influential$observation,]
remove.dffits <- data.frame(remove.dffits)
Train.clean.NO <- lm(data=remove.dffits, logSalePrice~ GrLivArea + HouseAge + TotalBsmtSF + QualityIndex + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class + i1 +i2 +i3 +i9+i10+i11 )
Train.clean.NO
summary(Train.clean.NO)
anova(Train.clean.NO)
par(mfrow=c(2,2))
plot(Train.clean.NO)

sort(vif(Train.clean.NO),decreasing=TRUE)

accuracy(Train.clean.NO)
test.clean$logSalePrice <- log(test.clean$SalePrice)
accuracy(test.clean$final,x=test.clean$logSalePrice)


test.clean$logfinal <- predict(Train.clean.NO,newdata=test.clean)
test.clean$final <- exp(test.clean$logfinal)
accuracy(test.clean$final,x=test.clean$SalePrice)

Train.clean$logfinal <- predict(Train.clean.NO,newdata=Train.clean)
Train.clean$final <- exp(Train.clean$logfinal)
accuracy(Train.clean$final,x=Train.clean$SalePrice)


qi.hood <-lm(data=Filter13, logSalePrice ~ d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class+ price_sqft )
summary(qi.hood)
par(mfrow=c(2,2))
plot(qi.hood)

qi.hood.int <-lm(data=Filter13, logSalePrice ~ d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class+ price_sqft + (d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class)*price_sqft)
summary(qi.hood.int)
par(mfrow=c(2,2))
plot(qi.hood.int)

anova(qi.hood,qi.hood.int)



close1 <- lm(data=Filter13, logSalePrice~ GrLivArea + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class+d1_KitchenQual+d2_KitchenQual+d3_KitchenQual + (d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class)*(d1_KitchenQual+d2_KitchenQual+d3_KitchenQual)+ price_sqft*(d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class))
summary(close1)
par(mfrow=c(2,2))
plot(close1)


close3 <- lm(data=Filter13, logSalePrice~ GrLivArea + HouseAge+ d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class +  d1_Zoning + d2_Zoning)
summary(close3)
par(mfrow=c(2,2))
plot(close3)

best <- lm(data=Filter13, logSalePrice~ GrLivArea + HouseAge+ d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class +  d1_Zoning + d2_Zoning)
summary(best)
par(mfrow=c(2,2))
plot(best)

ddply(Filter13, .(Neighborhood.class), summarize,  Mean.SalePrice=mean(SalePrice), Median.SalePrice=median(SalePrice), sd.SalePrice=sd(SalePrice), min.SalePrice=min(SalePrice), max.SalePrice=max(SalePrice))

ggplot(Filter13, aes(x=Neighborhood.class, y=SalePrice)) +
  geom_boxplot(fill="blue") +
  labs(title="Boxplot of Sale Price by Neighborhood.class") +
  theme(plot.title=element_text(lineheight=0.8, face="bold", hjust=0.5))

### TEST/TRAIN SPLIT 70/30


# Set the seed on the random number generator so you get the same split every time that you run the code.
set.seed(123)
Filter13$u <- runif(n=dim(Filter13)[1],min=0,max=1);

# Create train/test split;
train.df <- subset(Filter13, u<0.70);
test.df  <- subset(Filter13, u>=0.70);

# Check your data split. The sum of the parts should equal the whole.
# Do your totals add up?
dim(Filter13)[1]
dim(train.df)[1]
dim(test.df)[1]
dim(train.df)[1]+dim(test.df)[1]



#Automated Variable Selection

#Trick 1 create DF with only variables desired 
Train.clean <- train.df[,c(6,40,45,48,58,64,82,84,85,87,88,96,97,98,100,101,102,103,106,107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125)]
Train.clean <- na.omit(Train.clean)

test.clean <- test.df[,c(6,40,45,48,58,64,82,84,85,87,88,96,97,98,100,101,102,103,106,107,108,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125)]
test.clean <- na.omit(test.clean)


#Trick 2 specify the upper model and lower models using these R shortcuts.

# Define the upper model as the FULL model
upper.lm <- lm(SalePrice ~ .,data=Train.clean);
summary(upper.lm)

# Define the lower model as the Intercept model
lower.lm <- lm(SalePrice ~ 1,data=Train.clean);
summary(lower.lm)

# Need a SLR to initialize stepwise selection
sqft.lm <- lm(SalePrice ~ GrLivArea,data=Train.clean);
summary(sqft.lm)


###Trick #3: use the R function formula() to pass your shortcut definition of the Full Model to the scope argument in stepAIC().

# Note: There is only one function for classical model selection in R - stepAIC();
# stepAIC() is part of the MASS library.
# The MASS library comes with the BASE R distribution, but you still need to load it;

library(MASS)

# Call stepAIC() for variable selection
forward.lm <- stepAIC(object=lower.lm,scope=list(upper=formula(upper.lm),lower=~1),
                      direction=c('forward'));
summary(forward.lm)
anova(forward.lm)
par(mfrow=c(2,2))
plot(forward.lm)

backward.lm <- stepAIC(object=upper.lm,direction=c('backward'));
summary(backward.lm)
anova(backward.lm)
par(mfrow=c(2,2))
plot(backward.lm)


stepwise.lm <- stepAIC(object=sqft.lm,scope=list(upper=formula(upper.lm),lower=~1),
                       direction=c('both'));
summary(stepwise.lm)
anova(stepwise.lm)
par(mfrow=c(2,2))
plot(stepwise.lm)


steplm <- lm(SalePrice ~ GrLivArea + price_sqft + d1_KitchenQual + d3_Neighborhood.class + 
               d2_Zoning + d1_Neighborhood.class + FirstFlrSF + Fireplaces + 
               Bathrooms + d3_KitchenQual + LotArea + GarageArea + d1_Zoning + 
               QualityIndex, data=Train.clean)
summary(steplm)
AIC(steplm)

junk.lm <- lm(SalePrice ~ GrLivArea + QualityIndex + price_sqft + TotalBsmtSF + HouseAge + GarageArea + FirstFlrSF + LotArea + Bathrooms + Fireplaces + d1_Neighborhood.class + d2_Neighborhood.class + d3_Neighborhood.class + d1_KitchenQual + d2_KitchenQual + d3_KitchenQual + d1_Zoning + d2_Zoning + d3_Zoning + d4_Zoning, data=Train.clean)
summary(junk.lm)
anova(junk.lm)
par(mfrow=c(2,2))
plot(junk.lm)


# Compute the VIF values
library(car)
sort(vif(forward.lm),decreasing=TRUE)
sort(vif(backward.lm),decreasing=TRUE)
sort(vif(stepwise.lm),decreasing=TRUE)
sort(vif(junk.lm),decreasing=TRUE)

#Accuracy Details
accuracy(forward.lm)
accuracy(backward.lm)
accuracy(stepwise.lm)
accuracy(junk.lm)

get_mse(forward.lm,var.estimate = FALSE)
get_mse(backward.lm,var.estimate = FALSE)
get_mse(stepwise.lm,var.estimate = FALSE)
get_mse(junk.lm,var.estimate = FALSE)

#Simple AIC
AIC(junk.lm)
AIC(stepwise.lm)

BIC(junk.lm)
BIC(stepwise.lm)

library(forecast)

forward.test <- predict(forward.lm,newdata=test.clean)
str(forward.test)
backward.test <- predict(backward.lm,newdata=test.clean)
stepwise.test <- predict(stepwise.lm,newdata=test.clean)
junk.test <- predict(junk.lm,newdata=test.clean)

accuracy(forward.test, x=test.clean$SalePrice)
accuracy(forward.lm)
accuracy(backward.test, x=test.clean$SalePrice)
accuracy(stepwise.test, x=test.clean$SalePrice)
accuracy(junk.test, x=test.clean$SalePrice)

library(dvmisc)
get_mse(forward.lm,var.estimate = FALSE)

### Testing - compute the adjusted R-Squared, AIC, BIC, mean squared error
par(mfrow=c(2,2))
plot(forward.lm)


#forward.lm
intercept <- -1.518e+05
b1 <- 1.069e+02 
b2 <- 1.349e+03
b3 <- 1.723e+04
b4 <- 8.185e+03
b5 <- 6.010e+00
b6 <- -5.338e+03
b7 <- -8.357e+01
b8 <- -7.245e+03 
b9 <- 2.762e+03
b10 <- 3.230e-01 
b11 <- -5.118e+03
b12 <- -2.996e+03
b13 <- 8.515e+03
b14 <- 3.340e+00
b15 <- -3.319e+03
b16 <- 1.294e+03

#y_hat
test.clean$forward.lm.prediction <- intercept + b1*test.clean$GrLivArea + b2*test.clean$price_sqft + b3*test.clean$d1_KitchenQual + b4*test.clean$d3_Neighborhood.class + b5*test.clean$TotalBsmtSF + b6*test.clean$d1_Neighborhood.class + b7*test.clean$HouseAge + b8*test.clean$d2_Zoning + b9*test.clean$Fireplaces + b10*test.clean$LotArea + b11*test.clean$d3_KitchenQual + b12*test.clean$d2_Neighborhood.class + b13*test.clean$d1_Zoning + b14*test.clean$GarageArea +b15*test.clean$d2_KitchenQual + b16*test.clean$Bathrooms 

test.clean$forward.lm.error <- test.clean$SalePrice - test.clean$forward.lm.prediction
test.clean$forward.lm.squared.error <- (test.clean$SalePrice - test.clean$forward.lm.prediction)**2
mean(test.clean$forward.lm.squared.error)
avg.SalePrice <- mean(test.df$SalePrice) #y_bar

test.df$ylessybar2 <- (test.df$SalePrice - avg.SalePrice)**2

SSError <- sum(test.df$forward.lm.error)
SSTotal <- sum(test.df$ylessybar2)

R.sq <- 1-(SSE/SSTotal)
R.sq
adjR.sq <- 1-((1-R.sq)*(840-1))/(840-14-1)
adjR.sq

#backward.lm

intercept <- -1.641e+05
b1 <- 1.363e-01
b2 <- 6.104e+00
b3 <- 1.038e+02 
b4 <- 2.421e+03
b5 <- 6.826e+00
b6 <- 1.387e+03
b7 <- 3.824e+03
b8 <- 1.920e+04
b9 <- -2.036e+03
b10 <- 8.346e+03 
b11 <- -8.246e+03 
b12 <- -3.876e+03
b13 <- -1.327e+03
b14 <- 1.041e+04

test.df$backward.lm.prediction <- intercept + b1*test.df$LotArea + b2*test.df$FirstFlrSF + b3*test.df$GrLivArea + b4*test.df$Fireplaces + b5*test.df$GarageArea + b6*test.df$price_sqft + b7*test.df$Bathrooms + b8*test.df$d1_KitchenQual + b9*test.df$d3_KitchenQual + b10*test.df$d1_Zoning + b11*test.df$d2_Zoning + b12*test.df$d1_Neighborhood.class + b13*test.df$d2_Neighborhood.class + b14*test.df$d3_Neighborhood.class 

test.df$backward.lm.error <- test.df$SalePrice - test.df$backward.lm.prediction
test.df$backward.lm.squared.error <- (test.df$SalePrice - test.df$backward.lm.prediction)**2

avg.SalePrice <- mean(test.df$SalePrice) #y_bar

test.df$ylessybar2 <- (test.df$SalePrice - avg.SalePrice)**2

SSError <- sum(test.df$backward.lm.squared.error)
SSTotal <- sum(test.df$ylessybar2)

#stepwise

#Same as other two since coefficients are the same

# junk

intercept <- -1.520e+05
b1 <- 1.064e+02
b2 <- 2.961e+01
b3 <- 1.343e+03
b4 <- 5.866e+00
b5 <- -9.126e+01
b6 <- 3.166e+00
b7 <- 3.983e-01
b8 <- 3.698e-01 
b9 <- 1.290e+03
b10 <- 2.766e+03
b11 <- -5.071e+03 
b12 <- -2.755e+03
b13 <- 8.475e+03
b14 <- 1.685e+04
b15 <- -3.637e+03
b16 <- -5.304e+03 
b17 <- 9.230e+03
b18 <- -7.045e+03
b19 <- 2.812e+03
b20 <- 1.094e+03

test.clean$junk.lm.prediction <- intercept + b1*test.clean$GrLivArea + b2*test.clean$QualityIndex + b3*test.clean$price_sqft + b4*test.clean$TotalBsmtSF + b5*test.clean$HouseAge + b6*test.clean$GarageArea + b7*test.clean$FirstFlrSF + b8*test.clean$LotArea + b9*test.clean$Bathrooms + b10*test.clean$Fireplaces + b11*test.clean$d1_Neighborhood.class + b12*test.clean$d2_Neighborhood.class + b13*test.clean$d3_Neighborhood.class + b14*test.clean$d1_KitchenQual + b15*test.clean$d2_KitchenQual + b16*test.clean$d3_KitchenQual + b17*test.clean$d1_Zoning + b18*test.clean$d2_Zoning + b19*test.clean$d3_Zoning +b20*test.clean$d4_Zoning 

test.clean$junk.lm.error <- test.clean$SalePrice - test.clean$junk.lm.prediction
test.clean$junk.lm.squared.error <- (test.clean$SalePrice - test.clean$junk.lm.prediction)**2
mean(test.clean$junk.lm.squared.error)

avg.SalePrice <- mean(test.df$SalePrice) #y_bar

test.df$ylessybar2 <- (test.df$SalePrice - avg.SalePrice)**2

SSError <- sum(test.df$junk.lm.squared.error)
SSTotal <- sum(test.df$ylessybar2)

R.sq <- 1-(SSError/SSTotal)
R.sq
adjR.sq <- 1-((1-R.sq)*(840-1))/(840-14-1)
adjR.sq


accuracy(forward.test, x=test.clean$SalePrice)
accuracy(backward.test, x=test.clean$SalePrice)
accuracy(stepwise.test, x=test.clean$SalePrice)
accuracy(junk.test, x=test.clean$SalePrice)

accuracy(forward.lm)
accuracy(backward.lm)
accuracy(stepwise.lm)
accuracy(junk.lm)

#####Operational Validation

#Forward Selection

# Training Data
# Abs Pct Error
forward.pct <- abs(forward.lm$residuals)/Train.clean$SalePrice;

# Assign Prediction Grades;
forward.PredictionGrade <- ifelse(forward.pct<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(forward.pct<=0.15,'Grade 2: (0.10,0.15]',
                                         ifelse(forward.pct<=0.25,'Grade 3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )					
)

forward.trainTable <- table(forward.PredictionGrade)
forward.trainTable/sum(forward.trainTable)


# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice-forward.test)/test.clean$SalePrice;
backward.testPCT <- abs(test.clean$SalePrice-backward.test)/test.clean$SalePrice;
stepwise.testPCT <- abs(test.clean$SalePrice-stepwise.test)/test.clean$SalePrice;
junk.testPCT <- abs(test.clean$SalePrice-junk.test)/test.clean$SalePrice;


# Assign Prediction Grades;
forward.testPredictionGrade <- ifelse(forward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                      ifelse(forward.testPCT<=0.15,'Grade 2: (0.10,0.15]',
                                             ifelse(forward.testPCT<=0.25,'Grade 3: (0.15,0.25]',
                                                    'Grade 4: (0.25+]')
                                      )					
)

forward.testTable <-table(forward.testPredictionGrade)
forward.testTable/sum(forward.testTable)


####backward selection

# Training Data
# Abs Pct Error
backward.pct <- abs(backward.lm$residuals)/Train.clean$SalePrice;

# Assign Prediction Grades;
backward.PredictionGrade <- ifelse(backward.pct<=0.10,'Grade 1: [0.0.10]',
                                  ifelse(backward.pct<=0.15,'Grade 2: (0.10,0.15]',
                                         ifelse(backward.pct<=0.25,'Grade 3: (0.15,0.25]',
                                                'Grade 4: (0.25+]')
                                  )					
)

backward.trainTable <- table(backward.PredictionGrade)
backward.trainTable/sum(backward.trainTable)


# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice-forward.test)/test.clean$SalePrice;
backward.testPCT <- abs(test.clean$SalePrice-backward.test)/test.clean$SalePrice;
stepwise.testPCT <- abs(test.clean$SalePrice-stepwise.test)/test.clean$SalePrice;
junk.testPCT <- abs(test.clean$SalePrice-junk.test)/test.clean$SalePrice;


# Assign Prediction Grades;
backward.testPredictionGrade <- ifelse(backward.testPCT<=0.10,'Grade 1: [0.0.10]',
                                      ifelse(backward.testPCT<=0.15,'Grade 2: (0.10,0.15]',
                                             ifelse(backward.testPCT<=0.25,'Grade 3: (0.15,0.25]',
                                                    'Grade 4: (0.25+]')
                                      )					
)

backward.testTable <-table(backward.testPredictionGrade)
backward.testTable/sum(backward.testTable)

#####stepwise

# Training Data
# Abs Pct Error
stepwise.pct <- abs(stepwise.lm$residuals)/Train.clean$SalePrice;

# Assign Prediction Grades;
stepwise.PredictionGrade <- ifelse(stepwise.pct<=0.10,'Grade 1: [0.0.10]',
                                   ifelse(stepwise.pct<=0.15,'Grade 2: (0.10,0.15]',
                                          ifelse(stepwise.pct<=0.25,'Grade 3: (0.15,0.25]',
                                                 'Grade 4: (0.25+]')
                                   )					
)

stepwise.trainTable <- table(stepwise.PredictionGrade)
stepwise.trainTable/sum(stepwise.trainTable)


# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice-forward.test)/test.clean$SalePrice;
backward.testPCT <- abs(test.clean$SalePrice-backward.test)/test.clean$SalePrice;
stepwise.testPCT <- abs(test.clean$SalePrice-stepwise.test)/test.clean$SalePrice;
junk.testPCT <- abs(test.clean$SalePrice-junk.test)/test.clean$SalePrice;


# Assign Prediction Grades;
stepwise.testPredictionGrade <- ifelse(stepwise.testPCT<=0.10,'Grade 1: [0.0.10]',
                                       ifelse(stepwise.testPCT<=0.15,'Grade 2: (0.10,0.15]',
                                              ifelse(stepwise.testPCT<=0.25,'Grade 3: (0.15,0.25]',
                                                     'Grade 4: (0.25+]')
                                       )					
)

stepwise.testTable <-table(stepwise.testPredictionGrade)
stepwise.testTable/sum(stepwise.testTable)

####junk


# Training Data
# Abs Pct Error
junk.pct <- abs(junk.lm$residuals)/Train.clean$SalePrice;

# Assign Prediction Grades;
junk.PredictionGrade <- ifelse(junk.pct<=0.10,'Grade 1: [0.0.10]',
                                   ifelse(junk.pct<=0.15,'Grade 2: (0.10,0.15]',
                                          ifelse(junk.pct<=0.25,'Grade 3: (0.15,0.25]',
                                                 'Grade 4: (0.25+]')
                                   )					
)

junk.trainTable <- table(junk.PredictionGrade)
junk.trainTable/sum(junk.trainTable)


# Test Data
# Abs Pct Error
forward.testPCT <- abs(test.clean$SalePrice-forward.test)/test.clean$SalePrice;
backward.testPCT <- abs(test.clean$SalePrice-backward.test)/test.clean$SalePrice;
stepwise.testPCT <- abs(test.clean$SalePrice-stepwise.test)/test.clean$SalePrice;
junk.testPCT <- abs(test.clean$SalePrice-junk.test)/test.clean$SalePrice;


# Assign Prediction Grades;
junk.testPredictionGrade <- ifelse(junk.testPCT<=0.10,'Grade 1: [0.0.10]',
                                       ifelse(junk.testPCT<=0.15,'Grade 2: (0.10,0.15]',
                                              ifelse(junk.testPCT<=0.25,'Grade 3: (0.15,0.25]',
                                                     'Grade 4: (0.25+]')
                                       )					
)

junk.testTable <-table(junk.testPredictionGrade)
junk.testTable/sum(junk.testTable)
