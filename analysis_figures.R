#title: Analysis and figures for Knight et al. Pre-processing spectrogram parameters improve the accuracy of bioacoustic classification using convolutional neural networks. Bioacoustics 10.1080/09524622.2019.1606734.
#author: Elly C. Knight
#date: May 14, 2017

library(tidyverse)
library(stringi)
library(MuMIn)
library(gridExtra)
library(agricolae)
library(nls2)
library(tuneR)
library(seewave)
library(reshape2)
library(signal)
library(warbleR)
library(MASS)
library(viridis)
library(plyr)


summarySE <- function(data=NULL, measurevar, groupvars=NULL, na.rm=FALSE,
                      conf.interval=.95, .drop=TRUE) {
  
  # New version of length which can handle NA's: if na.rm==T, don't count them
  length2 <- function (x, na.rm=FALSE) {
    if (na.rm) sum(!is.na(x))
    else       length(x)
  }
  
  # This does the summary. For each group's data frame, return a vector with
  # N, mean, and sd
  datac <- ddply(data, groupvars, .drop=.drop,
                 .fun = function(xx, col) {
                   c(N    = length2(xx[[col]], na.rm=na.rm),
                     mean = mean   (xx[[col]], na.rm=na.rm),
                     sd   = sd     (xx[[col]], na.rm=na.rm)
                   )
                 },
                 measurevar
  )
  
  # Rename the "mean" column    
  datac <- plyr::rename(datac, c("mean" = measurevar))
  
  datac$se <- datac$sd / sqrt(datac$N)  # Calculate standard error of the mean
  
  # Confidence interval multiplier for standard error
  # Calculate t-statistic for confidence interval: 
  # e.g., if conf.interval is .95, use .975 (above/below), and use df=N-1
  ciMult <- qt(conf.interval/2 + .5, datac$N-1)
  datac$ci <- datac$se * ciMult
  
  return(datac)
}

my.theme <- theme_classic() +
  theme(text=element_text(size=16, family="Arial"),
        axis.text.x=element_text(size=12),
        axis.text.y=element_text(size=12),
        axis.line.x=element_line(linetype=1),
        axis.line.y=element_line(linetype=1),
        axis.title.x=element_text(margin=margin(10,0,0,0), size=18),
        axis.title.y=element_text(margin=margin(0,10,0,0), size=18))


#1. Frequency characterization----

#1a. Extract frequencies----
species <- c("ALFL", "AMRE", "BADO", "BBWA", "BCCH",
             "BRCR", "CONI", "CORA", "GHOW", "HETH",
             "LEFL", "OSFL", "REVI", "SWTH", "TEWA",
             "WBNU", "WTSP", "YRWA", "YWAR")
freqmin <- c(1.8, 3, 0.1, 5, 2,
             3, 2, 1, 0.2, 3,
             3, 2, 2, 2, 3,
             1, 1.5, 3, 3)
freqmax <- c(6.4, 8, 0.5, 9, 7,
             8.5, 5, 3, 0.4, 7,
             7, 4, 6, 6, 7,
             3, 6.6, 6, 8)

species <- c("YWAR")
freqmin <- c(3)
freqmax <- c(8)

data.freq2 <- data.frame()
for (i in 1:length(species)) {
  setwd(dir=paste0("/Users/ellyknight/Google Drive/step12_clean/",species[i]))
  filelist <- as.data.frame(list.files(pattern="*.wav$", recursive=TRUE))
  colnames(filelist) <- c("Fname")
  
  sound.files <- as.character(filelist$Fname)
  channel <- rep(1, length(sound.files))
  selec <- rep(1, length(sound.files))
  start <- rep(0, length(sound.files))
  
  end <- numeric()
  for(j in 1:length(sound.files)){
    file <- sound.files[j]
    data <- readWave(filename = file)
    end.j <- round(length(data@left)/data@samp.rate, 2)
    end <- rbind(end, end.j)
  }
  
  select.table <- data.frame(sound.files, channel, selec, start, end)
  str(select.table)
  dat1 <- dfts(X=select.table, length.out=100, bp=c(0.1,9), threshold=30)
  dat2 <- dat1 %>% 
    gather(key=sample, value=dfreq, c(3:102))
  dat3 <- dat2 %>% 
    dplyr::filter(sample != "selec") %>% 
    dplyr::filter(dfreq > 0) %>% 
    group_by(sound.files) %>% 
    summarize(dfreq.mn = mean(dfreq),
              dfreq.min = min(dfreq),
              dfreq.max = max(dfreq)) %>% 
    mutate(species=paste0(species[i]))
  data.freq2 <- rbind(data.freq2, dat3)
  message('Completed',species[i])
  
}


setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis")
#write.csv(data.freq, "WarbleRFrequencyExtraction.csv", row.names=FALSE)

#1b. Summary of species for table 2----
data.freq <- read.csv("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/WarbleRFrequencyExtraction.csv")

data.sum <- data.freq %>% 
  dplyr::group_by(species) %>% 
  dplyr::summarize(mn=mean(dfreq.mn),
                   sd=sd(dfreq.mn)) %>% 
  ungroup() %>% 
  arrange(mn)


#2. General parameters----

#2a. Wrangling----
setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/OriginalExperiment")

#Get file list
file_list <- list.files()

#Merge function
for (i in 1:length(file_list)){
  
  # if the merged dataset doesn't exist, create it
  if (!exists("dataset")){
    dataset <- read.csv(file_list[i], header=TRUE)
    dataset$det.file <- file_list[i]
  }
  else
    # if the merged dataset does exist, append to it
  {
    temp_dataset <-read.csv(file_list[i], header=TRUE)
    temp_dataset$det.file <- file_list[i]
    dataset<-rbind(dataset, temp_dataset)
    rm(temp_dataset)
  }
}

log4 <- dataset %>% 
  separate(det.file, into=c("dim", "data", "clean", "net", "epochs", "batch", "trial", "scale"), sep="_") %>% 
  dplyr::select(-Avg.Accuracy, -Standard.Deviation, -dim, -data, -net, -epochs, -batch, -trial) %>% 
  gather(trial, accuracy, Trial.1:Trial.4) %>% 
  arrange(scale, FFT.Buckets, Time.Window.Size, trial) %>% 
  mutate(trial=as.numeric(stri_sub(trial, 7, 7)),
         accuracy=as.numeric(stri_sub(accuracy, 1, 4))/100,
         percent=accuracy*100)  %>% 
  mutate(scale=case_when(scale == "spectrogramlinearBucketsLinearAmp.csv"
                        ~ "Linear - dB",
                        scale == "spectrogramlinearBucketsLogAmp.csv"
                        ~ "Linear - log dB",
                        scale == "spectrogramlogBucketsLinearAmp.csv"
                        ~ "Log - dB",
                        scale == "spectrogramlogBucketsLogAmp.csv"
                        ~ "Log - log dB",
                        scale == "spectrogramnormal.csv"
                        ~ "Composite")) %>% 
  dplyr::rename(size=FFT.Buckets, length=Time.Window.Size) %>% 
  separate(scale, into=c("freq", "amp"), sep=" - ", remove=FALSE) %>% 
  mutate(freq = ifelse(freq=="2x2", NA, freq),
         length = length*1000)

#3b. Model----
log5 <- log4 %>% 
  dplyr::filter(scale != "Composite")

lm1 <- lm(percent ~ freq + amp + size + length + I(length^2), data=log5, na.action="na.fail")
summary(lm1)
dredge(lm1)

#3c. Validation----

#homogeneity
lm.E <- resid(lm1)
lm.F <- fitted(lm1)
par(mfrow=c(1,1))
plot(x = lm.F,
     y = lm.E,
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, v = 0, lty = 2)

#influence
par(mfrow = c(1, 1))
plot(cooks.distance(lm1), type = "h", ylim = c(0, 1))
abline(h = 1)

#Normality
hist(lm.E, breaks = 15)

#3d. Plot----
newdata <- expand.grid(freq = c("Linear"), amp = c("log dB"), length = seq(min(log5$length), max(log5$length), 0.1), size = mean(log5$size))
newdata1 <- as.data.frame(predict(lm1, newdata, se.fit=TRUE, type="response")) %>% 
  mutate(lwr=fit-1.96*se.fit, upr=fit+1.96*se.fit) %>% 
  cbind(newdata) %>% 
  arrange(desc(fit))
head(newdata1)

newdata <- expand.grid(freq = c("Linear"), amp = c("log dB"), size = seq(min(log5$size), max(log5$size), 1), length = mean(log5$length))
newdata2 <- as.data.frame(predict(lm1, newdata, se.fit=TRUE, type="response")) %>% 
  mutate(lwr=fit-1.96*se.fit, upr=fit+1.96*se.fit) %>% 
  cbind(newdata) %>% 
  arrange(desc(fit))
head(newdata2)

plot.size <- ggplot() +
  geom_ribbon(aes(x=size, ymin=lwr, ymax=upr), alpha=0.15, col="grey", data=newdata2) +
  geom_line(aes(x=size, y=fit), lwd=1.2, linejoin="round", data=newdata2) +
  coord_cartesian(ylim=c(86,98)) +
  scale_y_continuous(breaks=c(88,90,92,94,96,98)) +
  geom_jitter(aes(x=size, y=percent, colour=length), data=log5, width=3) +
  xlab("Number of frequency segments") +
  ylab("") +
  scale_colour_viridis_c(name = "Window\nlength (ms)") +
  my.theme
plot.size

plot.length <- ggplot() +
  geom_ribbon(aes(x=length, ymin=lwr, ymax=upr), alpha=0.15, col="grey", data=newdata1) +
  geom_line(aes(x=length, y=fit), lwd=1.2, linejoin="round", data=newdata1) +
  coord_cartesian(ylim=c(86,98)) +
  scale_y_continuous(breaks=c(88,90,92,94,96,98)) +
  geom_jitter(aes(x=length, y=percent, colour=size), data=log5, width=3) +
  xlab("FFT window length (ms)") +
  ylab("Classification accuracy (%)") +
  scale_colour_viridis_c(name = "Number of\nfrequency\nsegments") +
  my.theme
plot.length

plot.resolution <- grid.arrange(plot.length, plot.size, ncol=2)

ggsave(plot=plot.resolution, "/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig3Resolution.png", device="png", width=14, height=6, dpi=600)


#4. Scale frequency regression----

#4a. Wrangling----

setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/confusionMatrices2")
files <- data.frame(list.files(path="/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/confusionMatrices2", recursive=TRUE))
colnames(files) <- "path"
files.list <- files %>% 
  separate(path, into=c("scale", "file"), sep="/", remove=FALSE) %>% 
  separate(file, into=c("fig", "data", "net", "epoch", "batch", "trial", "type", "test"), sep=
             "_") %>% 
  mutate(ID=paste0(scale, "-", trial)) %>% 
  mutate(scale = gsub("([[:lower:]])([[:upper:]])", "\\1 \\2", scale)) %>% 
  separate(scale, into=c("freq", "junk1", "amp", "junk2"), remove=FALSE) %>% 
  mutate(freq = ifelse(freq=="2x2", NA, freq))

confus.dat <- data.frame()
for (i in 1:nrow(files.list)){
  file.i <- paste0(files.list$path[i])
  dat <- read.csv(file.i, header=FALSE)
  colnames(dat) <- c("ALFL", "AMRE", "BADO", "BBWA", "BCCH",
                     "BRCR", "CONI", "CORA", "GHOW", "HETH",
                     "LEFL", "OSFL", "REVI", "SWTH", "TEWA",
                     "WBNU", "WTSP", "YRWA", "YWAR")
  rownames(dat) <- c("ALFL", "AMRE", "BADO", "BBWA", "BCCH",
                     "BRCR", "CONI", "CORA", "GHOW", "HETH",
                     "LEFL", "OSFL", "REVI", "SWTH", "TEWA",
                     "WBNU", "WTSP", "YRWA", "YWAR")
  dat <- as.matrix(dat)
  dat.long <- melt(dat) %>% 
    mutate(trial=paste0(files.list$trial[i]),
           scale=paste0(files.list$scale[i]),
           freq=paste0(files.list$freq[i]),
           amp=paste0(files.list$amp[i])) %>% 
    dplyr::rename(pred=Var1, species=Var2, acc=value)
  confus.dat <- rbind(confus.dat, dat.long)
}

confus <- confus.dat %>% 
  dplyr::filter(pred==species,
                scale!="2x2") %>% 
  left_join(data.sum, by="species") %>% 
  dplyr::rename(dfreq=mn, accuracy=acc) %>% 
  mutate(percent = accuracy*100)

#4b. Model----
lm2 <- lm(percent~dfreq*freq, data=confus, na.action = "na.fail")
summary(lm2)
dredge(lm2)

#4c. Validation----

#homogeneity
lm.E <- resid(lm2)
lm.F <- fitted(lm2)
par(mfrow=c(1,1))
plot(x = lm.F,
     y = lm.E,
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, v = 0, lty = 2)

#influence
par(mfrow = c(1, 1))
plot(cooks.distance(lm2), type = "h", ylim = c(0, 1))
abline(h = 1)

#Normality
hist(lm.E, breaks = 15)

#4d. Plot----

newdata1 <- expand.grid(freq=unique(confus$freq), dfreq = seq(min(confus$dfreq), max(confus$dfreq), 0.1))
newdata <- as.data.frame(predict(lm2, newdata1, se.fit=TRUE, type="response")) %>% 
  mutate(lwr=fit-1.96*se.fit, upr=fit+1.96*se.fit) %>% 
  cbind(newdata1)

cols <- viridis(4)[c(1,3)]

ggplot(data=newdata) +
  geom_ribbon(aes(x=dfreq, ymin=lwr, ymax=upr, group=freq), alpha=0.15, col="grey") +
  geom_line(aes(x=dfreq, y=fit, colour=freq), lwd=1.2, linejoin="round") +
  geom_jitter(aes(x=dfreq, y=percent, colour=freq), data=confus, width = 0.2, height = 1) +
  scale_y_continuous(limits=c(75,100)) +
  my.theme+
  labs(x="Mean dominant frequency (kHz)", y="Classification accuracy (%)") +
  scale_colour_manual(breaks=c("linear", "log"),
                      values=cols,
                      labels=c("Linear", "Log"),
                      name="Spectrogram\nfrequency\nscale")

ggsave("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig4Frequency.png", device="png", width=7.5, height=6, dpi=600)


#5. Composite comparison----

table(log4$size, log4$length, log4$scale)

#5a. Model---- 
lm3 <- lm(percent ~ scale + size + stats::poly(length,2), data=log4)
summary(lm3)

#5b. Validation----

#homogeneity
lm.E <- resid(lm3)
lm.F <- fitted(lm3)
par(mfrow=c(1,1))
plot(x = lm.F,
     y = lm.E,
     xlab = "Fitted values",
     ylab = "Residuals")
abline(h = 0, v = 0, lty = 2)

#influence
par(mfrow = c(1, 1))
plot(cooks.distance(lm3), type = "h", ylim = c(0, 1))
abline(h = 1)

#Normality
hist(lm.E, breaks = 15)

#5c. Scale type plot----
lm3 <- lm(percent ~ scale + stats::poly(log4$size, degree=1) + stats::poly(log4$length, degree=2), data=log4)
summary(lm3)

log4c <- summarySE(log4, measurevar="percent", groupvars="scale")
log4c

cols <- viridis(5)

ggplot(log4c, aes(x=scale, y=percent)) +
  geom_bar(aes(fill=scale), position=position_dodge(), stat="identity", show.legend = FALSE) +
  geom_errorbar(aes(ymin=percent-ci, ymax=percent+ci), width=0.2, position=position_dodge(.9)) +
  coord_cartesian(ylim=c(92,96)) +
  geom_text(aes(x=scale, y=percent+ci, label=c("a", "ab", "a", "b", "ab"), 
                vjust=-1), 
            position = position_dodge(width=1)) +
  xlab("Spectrogram scale (frequency - amplitude)") +
  ylab("Classification accuracy (%)") +
  scale_fill_manual(values=cols) +
  my.theme

ggsave("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig5Scale.png", device="png", width=6, height=6, dpi=600)

#5d. Summary parameters----
log4.sum <- log4 %>% 
  group_by(size, length, scale) %>% 
  dplyr::summarize(mean = mean(percent),
                   sd = sd(percent),
                   se = sd(percent)/sqrt(n())) %>% 
  arrange(desc(mean))
head(log4.sum)
tail(log4.sum)

summary(log4$percent)

best <- log4.sum %>% 
  dplyr::filter(scale=="Composite",
                size==113,
                length==50)
best
worst <- log4.sum %>% 
  dplyr::filter(scale=="Log - dB",
                size == 10,
                length == 0.5)
worst


#6. Frequency classification----

#6a. LDA
setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis")
data.freq <- read.csv("WarbleRFrequencyExtraction.csv")

n=10

data.lda <- data.frame()
confus.lda <- data.frame()
for (i in 1:n) {
  rep <- paste0(i)
  ind <- sample(1:nrow(data.freq), 0.75 * nrow(data.freq), replace=FALSE)
  train <- data.freq[ind,]
  test <- data.freq[-ind,]
  lda1 <- lda(species ~dfreq.mn, data=train)
  pred1 <- predict(lda1, newdata=test)
  pred <- pred1$class
  test1 <- cbind(pred, test) %>% 
    mutate(acc = ifelse(pred==species, 1, 0)) %>% 
    mutate(trial=rep)
  data.lda <- rbind(data.lda, test1)
  samples <- test %>% 
    group_by(species) %>% 
    dplyr::summarize(sample=n())
  confus <- test1 %>% 
    group_by(species, pred) %>% 
    dplyr::summarize(count=n()) %>% 
    left_join(samples, by="species") %>% 
    mutate(percent=count/sample,
           trial=rep) %>% 
    ungroup()
  confus.lda <- rbind(confus.lda, confus)
}

#6b. Wrangling----
lda.sum <- data.lda %>% 
  group_by(trial) %>% 
  dplyr::summarize(acc=sum(acc)/n()*100)
mean(lda.sum$acc)
sd(lda.sum$acc)
sd(lda.sum$acc)/sqrt(nrow(lda.sum))

samples <- data.frame(species=c("GHOW", "BADO", "CORA", "WBNU", "OSFL",
                                "CONI", "REVI", "SWTH", "WTSP", "ALFL",
                                "BCCH", "HETH", "YRWA", "LEFL", "TEWA",
                                "YWAR", "BRCR", "AMRE", "BBWA"),
                      samples=c(134, 139, 392, 97, 159,
                                100, 111, 120, 114, 133,
                                111, 293, 141, 491, 111,
                                116, 108, 103, 99),
                      order=c(1:19))

allcombos <- expand.grid(unique(as.factor(confus.lda$species)), unique(as.factor(confus.lda$species))) %>% 
  dplyr::rename(pred=Var1, species=Var2)

lda.confus.all <- full_join(confus.lda, allcombos, by=c("species", "pred")) %>% 
  mutate(percent=ifelse(is.na(percent), 0, percent)) %>% 
  dplyr::group_by(species, pred) %>%
  dplyr::summarize(mean=mean(percent))

lda.confus.sum <- lda.confus.all %>% 
  dplyr::filter(species==pred) %>% 
  arrange(mean)


#7. Confusion matrices----

#7a. Wrangle----
head(lda.confus.all, 2)
head(confus.dat, 2)

confus.both.lda <- lda.confus.all %>% 
  mutate(accuracy = 100*mean) %>% 
  dplyr::select(species, pred, accuracy) %>%
  mutate(class = "lda") %>% 
  ungroup()

confus.both.cnn <- confus.dat %>% 
  dplyr::filter(scale=="2x2") %>% 
  group_by(species, pred) %>% 
  dplyr::summarize(accuracy = mean(acc)*100) %>% 
  mutate(class = "cnn") %>% 
  ungroup()

confus.both.lda$species = factor(confus.both.lda$species, levels = c("BBWA", "AMRE", "BRCR",
                                                                     "TEWA", "YWAR", "LEFL",
                                                                     "YRWA", "HETH", "BCCH",
                                                                     "ALFL", "WTSP", "CONI",
                                                                     "REVI", "SWTH", "OSFL",
                                                                     "WBNU", "CORA", "GHOW",
                                                                     "BADO"),
                                 labels=c("Bay-breasted warbler",
                                          "American redstart",
                                          "Brown creeper",
                                          "Tennessee warbler",
                                          "Yellow warbler",
                                          "Least flycatcher",
                                          "Yellow-rumped warbler",
                                          "Hermit thrush",
                                          "Black-capped chickadee",
                                          "Alder flycatcher",
                                          "White-throated sparrow",
                                          "Common nighthawk",
                                          "Red-eyed vireo",
                                          "Swainson's thrush",
                                          "Olive-sided flycatcher",
                                          "White-breasted nuthatch",
                                          "Common raven",
                                          "Great horned owl",
                                          "Barred owl"))
confus.both.lda$pred = factor(confus.both.lda$pred, levels = c("BADO", "GHOW", "CORA", 
                                                               "WBNU", "OSFL", "SWTH",
                                                               "REVI", "CONI", 
                                                               "WTSP", "ALFL",
                                                               "BCCH", "HETH", "YRWA",
                                                               "LEFL", "YWAR", "TEWA", 
                                                               "BRCR", "AMRE",
                                                               "BBWA"),
                              labels=c("Barred owl",
                                       "Great horned owl",
                                       "Common raven",
                                       "White-breasted nuthatch",
                                       "Olive-sided flycatcher",
                                       "Swainson's thrush",
                                       "Red-eyed vireo",
                                       "Common nighthawk",
                                       "White-throated sparrow",
                                       "Alder flycatcher",
                                       "Black-capped chickadee",
                                       "Hermit thrush",
                                       "Yellow-rumped warbler",
                                       "Least flycatcher",
                                       "Yellow warbler",
                                       "Tennessee warbler",
                                       "Brown creeper",
                                       "American redstart",
                                       "Bay-breasted warbler"))
confus.both.cnn$species = factor(confus.both.cnn$species, levels = c("BBWA", "AMRE", "BRCR",
                                                                     "TEWA", "YWAR", "LEFL",
                                                                     "YRWA", "HETH", "BCCH",
                                                                     "ALFL", "WTSP", "CONI",
                                                                     "REVI", "SWTH", "OSFL",
                                                                     "WBNU", "CORA", "GHOW",
                                                                     "BADO"),
                                 labels=c("Bay-breasted warbler",
                                          "American redstart",
                                          "Brown creeper",
                                          "Tennessee warbler",
                                          "Yellow warbler",
                                          "Least flycatcher",
                                          "Yellow-rumped warbler",
                                          "Hermit thrush",
                                          "Black-capped chickadee",
                                          "Alder flycatcher",
                                          "White-throated sparrow",
                                          "Common nighthawk",
                                          "Red-eyed vireo",
                                          "Swainson's thrush",
                                          "Olive-sided flycatcher",
                                          "White-breasted nuthatch",
                                          "Common raven",
                                          "Great horned owl",
                                          "Barred owl"))
confus.both.cnn$pred = factor(confus.both.cnn$pred, levels = c("BADO", "GHOW", "CORA", 
                                                               "WBNU", "OSFL", "SWTH",
                                                               "REVI", "CONI", 
                                                               "WTSP", "ALFL",
                                                               "BCCH", "HETH", "YRWA",
                                                               "LEFL", "YWAR", "TEWA", 
                                                               "BRCR", "AMRE",
                                                               "BBWA"),
                              labels=c("Barred owl",
                                       "Great horned owl",
                                       "Common raven",
                                       "White-breasted nuthatch",
                                       "Olive-sided flycatcher",
                                       "Swainson's thrush",
                                       "Red-eyed vireo",
                                       "Common nighthawk",
                                       "White-throated sparrow",
                                       "Alder flycatcher",
                                       "Black-capped chickadee",
                                       "Hermit thrush",
                                       "Yellow-rumped warbler",
                                       "Least flycatcher",
                                       "Yellow warbler",
                                       "Tennessee warbler",
                                       "Brown creeper",
                                       "American redstart",
                                       "Bay-breasted warbler"))

#7b. Plot----

my.theme <- theme_classic() +
  theme(text=element_text(size=12, family="Arial"),
        axis.text.x=element_text(size=10),
        axis.text.y=element_text(size=10),
        axis.line.x=element_line(linetype=1),
        axis.line.y=element_line(linetype=1),
        axis.title.x=element_text(margin=margin(10,0,0,0), size=12),
        axis.title.y=element_text(margin=margin(0,10,0,0), size=12),
        legend.title=element_text(size = 12))


plot.cnn <- ggplot(confus.both.cnn) +
  geom_raster(aes(x=pred, y=species, fill=accuracy)) +
  scale_fill_viridis_c(name="AlexNet\nclassification\nrate(%)",
                       limits=c(0,10), breaks=c(0,5,10)) +
  ylab("True species") +
  xlab("Predicted species") +
  my.theme +
  theme(axis.text.x=element_text(angle=90, hjust=0.95, vjust=0.2))
plot.cnn

plot.lda <- ggplot(confus.both.lda) +
  geom_raster(aes(x=pred, y=species, fill=accuracy)) +
  scale_fill_viridis_c(name="LDA\nclassification\nrate (%)",
                       limits=c(0,100), breaks=c(0,50,100)) +
  ylab("True species") +
  xlab("Predicted species") +
  my.theme +
  theme(axis.text.x=element_text(angle=90, hjust=0.95, vjust=0.2))
plot.lda

png("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig6Confusion.png",
     width=14, height = 5.75, units="in",
     res=600)
grid.arrange(plot.cnn, plot.lda, ncol=2)
dev.off()

#8. Figure 2----
library(png)

setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig2")
files <- list.files(pattern=".png")

df <- data.frame(expand.grid(time=c("0.5", "1", "5", "10", "50", "100"),
                             freq=c("10", "25", "50", "75", "113"))) %>% 
  mutate(val=0)

emptyplot <- ggplot(df) +
  geom_raster(aes(x=time, y=freq, fill=val), show.legend=FALSE) +
  scale_x_discrete(labels=c("0.5", "1", "5", "10", "50", "100")) +
  scale_y_discrete(labels=c("10", "25", "50", "75", "113")) +
  scale_fill_gradient2(low="white", high="white") +
  xlab("FFT window length (ms)") +
  ylab("Number of frequency segments") + 
  my.theme

ggsave(plot=emptyplot,
       filename="/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig2EmptyPlot.png",
       device="png",
       dpi=600,
       width=6,
       height=5,
       units="in")

#9. Table 1----
#9a. Spectrograms----

setwd("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Table1")
files <- list.files(pattern=".wav")

for(i in 1:length(files)){
  file <- files[i]
  species <- str_sub(file, -11, -8)
  wav <- readWave(file)
  wav.f <- wav
  #if(species%in%c("CORA", "GHOW", "BADO")){
  #  wav.f <- bwfilter(wav, from=300, to=10000, output="Wave")
  #}
  #else
  #{
  #  wav.f <- bwfilter(wav, from=1000, to=10000, output="Wave")
  #}
  filename <- paste0(species,".png")
  png(filename,
      width=4, height = 2, units="in",
      res=600)
  par(mfrow=c(1,1), oma=c(0,0,0.5,0.5), mar=c(2,2,0,0))
  spectro(wav.f, grid=F, scale=F,
          flim=c(0,10),
          #        palette=function(x)rev(gray.colors(x)),
          collevels=seq(-30,0), ovlp=50,
          wl=1024,
          axisX=TRUE)
  dev.off()
}

#9b. Clip length----

species <- c("ALFL", "AMRE", "BADO", "BBWA", "BCCH",
             "BRCR", "CONI", "CORA", "GHOW", "HETH",
             "LEFL", "OSFL", "REVI", "SWTH", "TEWA",
             "WBNU", "WTSP", "YRWA", "YWAR")

cliplength <- data.frame()
for (i in 1:length(species)) {
  setwd(dir=paste0("/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Clips/",species[i]))
  filelist <- as.data.frame(list.files(pattern="*.wav$", recursive=TRUE))
  colnames(filelist) <- c("File")

  for(j in 1:nrow(filelist)){
    file <- as.character(filelist$File[j])
    data <- readWave(filename = file)
    filelist$Length[[j]] <- seewave::duration(data)
  }
  
  filelist$Species <- species[i]
  cliplength <- rbind(cliplength, filelist)
  
}

ggplot(cliplength) +
  geom_violin(aes(x=Species, y=Length))

cliplengthsummary <- cliplength %>% 
  dplyr::group_by(Species) %>% 
  dplyr::summarize(mean=mean(Length),
            sd=sd(Length))

#10. Investigate accuracy ~ clip length----
cliplengthanalysis <- confus.both.cnn %>% 
  dplyr::filter(species==pred) %>% 
  dplyr::mutate(Species=c("ALFL", "AMRE", "BADO", "BBWA", "BCCH",
                          "BRCR", "CONI", "CORA", "GHOW", "HETH",
                          "LEFL", "OSFL", "REVI", "SWTH", "TEWA",
                          "WBNU", "WTSP", "YRWA", "YWAR")) %>% 
  left_join(cliplengthsummary, by="Species") %>% 
  dplyr::select(-species, -pred) %>% 
  dplyr::rename(species=Species, length.mn = mean, length.sd = sd) %>%
  left_join(data.sum, by="species") %>% 
  dplyr::rename(freq.mn = mn, freq.sd = sd)

head(cliplengthanalysis)

ggplot(cliplengthanalysis) +
  geom_smooth(aes(x=length.mn, y=accuracy), method="lm") +
  geom_point(aes(x=length.mn, y=accuracy))

ggplot(cliplengthanalysis) +
  geom_smooth(aes(x=length.sd, y=accuracy), method="lm") +
  geom_point(aes(x=length.sd, y=accuracy))

ggplot(cliplengthanalysis) +
  geom_smooth(aes(x=freq.mn, y=accuracy), method="lm") +
  geom_point(aes(x=freq.mn, y=accuracy))

ggplot(cliplengthanalysis) +
  geom_smooth(aes(x=freq.sd, y=accuracy), method="lm") +
  geom_point(aes(x=freq.sd, y=accuracy))

ggplot(cliplengthanalysis) +
  geom_smooth(aes(x=freq.mn, y=length.mn), method="lm") +
  geom_point(aes(x=freq.mn, y=length.mn))

lm1 <- lm(accuracy ~ freq.mn, data=cliplengthanalysis)
summary(lm1)

lm2 <- lm(accuracy ~ length.mn, data=cliplengthanalysis)
summary(lm2)

lm3 <- lm(freq.mn ~ length.mn, data=cliplengthanalysis)
summary(lm3)

#10. Class accuracy values----
confus.dat.sp <- confus.both.cnn %>% 
  dplyr::filter(pred==species)

#11. Blank axes for figure 1----
df <- data.frame(expand.grid(time=seq(0,1, 0.1),
                             freq=seq(0,10,1))) %>% 
  mutate(val=0)

emptyplot <- ggplot(df) +
  geom_point(aes(x=time, y=freq, fill=val), colour="white", show.legend=FALSE) +
  scale_x_continuous(breaks=c(0,1), labels=c()) +
  scale_y_continuous(breaks=c(0,10), labels=c("0", "10")) +
  scale_fill_gradient2(low="white", high="white") +
  xlab("Time (s)") +
  ylab("Frequency (kHz)") + 
  my.theme + 
  theme(axis.ticks.x = element_blank())
emptyplot

ggsave(plot=emptyplot,
       filename="/Users/ellyknight/Documents/UoA/Projects/Projects/2x2/Analysis/Figs/Fig1EmptyPlot.png",
       device="png",
       dpi=600,
       width=8,
       height=2,
       units="in")
