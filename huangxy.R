  require(MASS)
  library(ISLR)
  library(boot)
  library(tree)
  library(rpart)
  library(rpart.plot)
  
  m<-matrix(0.95, nrow = 5, ncol = 5)
  diag(m)<-1
  n.train<-30
  
  x<-mvrnorm(30,mu=c(0,0,0,0,0),Sigma = m,empirical = TRUE)
  y<-rep(0,30)
  s1<-sample(which(x[,1]<=0.5),n.train*(2/3)*0.2)
  s2<-sample(which(x[,1]>0.5),n.train*(1/3)*0.8)
  y[s1]<-1;y[s2]<-1;
  y<-as.factor(y)
  
  n.test<-2000
  x.test<-mvrnorm(n.test,mu=c(0,0,0,0,0),Sigma = m,empirical = TRUE)
  y.test<-rep(0,n.test)
  s1.test<-sample(which(x.test[,1]<=0.5),n.test*(2/3)*0.2)
  s2.test<-sample(which(x.test[,1]>0.5),n.test*(1/3)*0.8)
  y.test[s1.test]<-1;y.test[s2.test]<-1
  y.test<-as.factor(y.test)
  
  par(mfrow=c(3,4))#ÉèÖÃÍ¼Æ¬´óÐ¡
  tree.original<-tree(y~.,data.frame(x,y))

  plot(tree.original)
  text(tree.original,pretty=0,all=TRUE)
  for(i in 1:11){
    tree.temp<-tree(y~.,data.frame(x,y),sample(1:30,30,replace=TRUE))
    plot(tree.temp)
    text(tree.temp)
  }
  
  
  
  tree.pred<-c()
  tree.pred.error.rate<-c()
  tree.prob.pred<-c()
  tree.prob.pred.error.rate<-c()
  tree.original.pred<-c()
  tree.original.pred.error.rate<-c()

  for (i in 1:200){
    tree.temp<-tree(y~.,data.frame(x,y)[sample(1:30,30,replace=TRUE),])###
    tree.pred.temp=predict(tree.temp,data.frame(x.test),type="class")
    tree.pred<-cbind(as.numeric(as.character(tree.pred.temp)),tree.pred)
    tree.pred.value<-ifelse(rowMeans(tree.pred)>0.5,1,0)
    table.temp<-table(as.factor(tree.pred.value),y.test)
    pred.error.rate<-(table.temp[2]+table.temp[3])/(table.temp[1]+table.temp[4])
    tree.pred.error.rate<-c(tree.pred.error.rate,pred.error.rate)
    
    
    y.prob<-as.numeric(as.character(y))
    tree.temp.prob<-tree(y.prob~.,data.frame(x,y.prob)[sample(1:30,30,replace=TRUE),])
    tree.pred.prob.temp=predict(tree.temp.prob,data.frame(x.test))
    tree.prob.pred<-cbind(tree.pred.prob.temp,tree.prob.pred)
    tree.pred.value.prob<-ifelse(rowMeans(tree.prob.pred)>0.5,1,0)
    table.prob.temp<-table(as.factor(tree.pred.value.prob),y.test)
    pred.prob.error.rate<-
      (table.prob.temp[2]+table.prob.temp[3])/(table.prob.temp[1]+table.prob.temp[4])
    tree.prob.pred.error.rate<-c(tree.prob.pred.error.rate,pred.prob.error.rate)
    
    tree.original<-tree(y~.,data.frame(x,y))
    tree.original.temp=predict(tree.original,data.frame(x.test),type="class")
    tree.original.pred<-cbind(as.numeric(as.character(tree.original.temp)),tree.original.pred)
    tree.original.value<-ifelse(rowMeans(tree.original.pred)>0.5,1,0)
    table.original.temp<-table(as.factor(tree.original.value),y.test)
    original.pred.error.rate<-
      (table.original.temp[2]+table.original.temp[3])/(table.original.temp[1]+table.original.temp[4])
    tree.original.pred.error.rate<-c(tree.original.pred.error.rate,original.pred.error.rate)
    
    
    
  }
  par(mfrow=c(1,1))
  
  plot(tree.pred.error.rate,ylim = c(0.15,0.6),col="orange",type="l",
       xlab="Number of Bootstrap Samples",ylab = "Test error")
  points(tree.pred.error.rate,col="orange")
  
  lines(tree.original.pred.error.rate,col="blue",type="l")
  points(tree.original.pred.error.rate,col="blue")
  
  lines(tree.prob.pred.error.rate,col="green")
  points(tree.prob.pred.error.rate,col="green")
  
  legend(x="topright",lty=c(1,1,1),
         legend=c("tree.original.pred.error.rate","tree.pred.error.rate",
                  "tree.prob.pred.error.rate"),col=c("blue","orange","green"))
  
  