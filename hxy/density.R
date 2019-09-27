library(quantmod)
library(zoo)
library(xts)
library(TTR)
library(tseries)
library(foreign)
library(KernSmooth)

Series<-getSymbols("AAPL",from="2018-01-01",to=Sys.Date(),src="yahoo")
AAPL_closing_price<-AAPL$AAPL.Close
plot(AAPL_closing_price)

close<-AAPL$AAPL.Close
close1<-lag(close,1)
calclose<-merge(close,close1)
simplerate<-(close-close1)/close1
names(simplerate)="simplerate"
calrate=merge(calclose,simplerate)
plot(simplerate)

simplerate1<-data.frame(simplerate)
simplerate1<-na.omit(simplerate1)
write.csv(simplerate1,file = "AAPL.csv")

plot(bkde(simplerate1[,1],kernel="normal"),type = "l",col="blue",xlab = 'Rate of return',ylab = 'Density',ylim = c(0,30))
lines(density(simplerate1[,1],kernel = c("triangular")),type = "l",col="green")
lines(density(simplerate1[,1],kernel = c("epanechnikov")),type = "l",col="red")
legend(x='topright',lty=c(1,1,1),
       legend=c("Gaunssian Kernel","Triangular Kernel",
                "Epanechnikov Kernel"),col=c("blue","green","red"))

plot(bkde(simplerate1[,1],kernel="normal"),type = "l",col="blue",xlab = 'Rate of return',ylab = 'Density',ylim = c(0,0.9))
lines(density(simplerate1[,1],kernel = c("triangular")),type = "l",col="green")
lines(density(simplerate1[,1],kernel = c("epanechnikov")),type = "l",col="red")

legend(x='bottom',lty=c(1,1,1),
       legend=c("Gaunssian Kernel","Triangular Kernel",
                "Epanechnikov Kernel"),col=c("blue","green","red"))
