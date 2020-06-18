# Code to run BSSANOVA on the examples from run-bssanova-compare.jl and run-bssanova-compare-gp.jl.

# Main BSSANOVA function, from the BSSANOVA.R code provided by Brian Reich at http://www4.stat.ncsu.edu/~reich/code/BSSANOVA.R
BSSANOVA<-function(y,x,categorical=NA,
    runs=10000,burn=2000,update=10,
    ae=.01,be=.01,priorprob=0.5,nterms=25,lambda=2,
    main=T,twoway=F,rest=F,const=10,fromfile=F,tag="results"){

#########  Definitions of the input    ################:
#  y is the n*1 vector of data
#  x is the n*p design matrix
#  categorical is the p-vector indicating (TRUE/FALSE) which
#       columns of x are categorical variables
#  runs is the number of MCMC samples to generate
#  burn is the number of samples to discard as burnin
#  update is the number of interations between displays
#  error variance sigma^2~Gamma(ae,be)
#  priorprob is the prior includion probability for 
#       each functional component
#  nterms is the number of eigenvector to retain
#  lambda is the hyperparameters in the half-Cauchy prior
#  main indicates whether to include main effects
#  twoway incicates whether to include interactions
#  rest indicates whether to include higher-order terms (f_0)
#  const is the relative variance for the polynomial trend

 
  #set some sample size parameters 
  n<-length(y)
  ncurves<-0
  p<-ncol(x)
  if(is.na(mean(categorical))){categorical<-rep(F,p)}

  if(main){ncurves<-ncurves+p}
  if(twoway){ncurves<-ncurves+p*(p-1)/2}
  if(rest){ncurves<-ncurves+1}
  term1<-term2<-rep(0,ncurves)
  if(p>0){o1<-order(x[,1])}
  if(p>1){o2<-order(x[,2])}
  if(p>2){o3<-order(x[,3])}
  if(p>3){o4<-order(x[,4])}

  #Set up the covariance matrices for the main effects
  Gamma<-array(0,c(ncurves,n,nterms))
  D<-array(0,c(nterms,ncurves))
  count<-1
  if(rest){totC<-matrix(0,n+nnew,n+nnew)}

  #set up the covariances for the main effects
  if(fromfile){
      if(main){for(j in 1:p){
         term1[count]<-term2[count]<-j
         if(rest){stop("Using rest with fromfile is not implemented.")}
         eig <- read.table(paste0(tag,"-eig-n=",n,"-j=",j,".dat"))
         Gamma[count,,]<-eig[2:nrow(eig),1:nterms];
         D[,count]<-abs(1/eig[1,1:nterms])
         count<-count+1
         print(j)
      }}
  } else {
      if(main){for(j in 1:p){
         term1[count]<-term2[count]<-j
         if(!categorical[j]){COV<-makeCOV.ME(x[,j],const=const)}
         if(categorical[j]){COV<-makeCOV.ME.cat(x[,j])}
         COV<-COV/mean(diag(COV[1:n,1:n]))
         if(rest){totC<-COV+totC}
         eig<-eigen(COV);
         Gamma[count,,]<-eig$vectors[,1:nterms];
         D[,count]<-abs(1/eig$values[1:nterms])
         count<-count+1
         write.table(rbind(eig$values[1:nterms],eig$vectors[,1:nterms]), file=paste0(tag,"-eig-n=",n,"-j=",j,".dat"), row.names=F, col.names=F)
         print(j)
      }}
  }

  #set up the covariances for the two-way interactions
  if(twoway){for(j1 in 2:p){for(j2 in 1:(j1-1)){
     term1[count]<-j1;term2[count]<-j2
     COV<-makeCOV.INT(x[,j1],x[,j2],const=const)
     COV<-COV/mean(diag(COV[1:n,1:n]))
     if(rest){totC<-COV+totC}
     eig<-eigen(COV);
     Gamma[count,,]<-eig$vectors[,1:nterms];
     D[,count]<-abs(1/eig$values[1:nterms])
    count<-count+1
  }}}

  #Set up the covariance matrices for the remainder part of the GP
  if(rest){
     term1[count]<-term2[count]<-1
     rCOV<-matrix(1,n+nnew,n+nnew)
     for(j in 1:p){rCOV<-rCOV*makeCOV.ME(c(x[,j],xnew[,j]),const=1)}
     COV<-rCOV-totC
     COV<-COV/mean(diag(COV[1:n,1:n]))
     eig<-eigen(COV);
     Gamma[count,,]<-eig$vectors[,1:nterms];
     D[,count]<-abs(1/eig$values[1:nterms])
  }

  ########                 Initial values      ###################
  int<-mean(y)
  sige<-sd(y);taue<-1/sige^2
  curves<-matrix(0,n,ncurves)
  curfit<-int+sige*apply(curves,1,sum)
  r<-rep(0,ncurves)

  #keep track of the mean of the fitted values
  afterburn<-0
  sumfit<-sumfit2<-rep(0,n)
  suminout<-rep(0,ncurves)
  sumcurves<-sumcurves2<-matrix(0,n,ncurves)
  keepr<-keepl2<-matrix(0,runs,ncurves)
  dev<-keepsige<-keepint<-rep(0,runs)
  ks<-rep(0,runs)
  slopes<-matrix(0,runs,ncurves)
  intercepts<-matrix(0,runs,ncurves)

  npts<-50
  if(priorprob==1){mxgx<-grid<-c(seq(0.0001,2,length=npts/2),seq(2,1000,length=npts/2))}
  if(priorprob<1){mxgx<-grid<-c(seq(0,2,length=npts/2),seq(2,1000,length=npts/2))}
 
  ########             Start the sampler       ###################
  countiter<-0
  start<-proc.time()[3]
  for(i in 1:runs){

   #new taue
    cantaue<-rnorm(1,taue,0.05*sd(y))
    if(cantaue>0){
      cansige<-1/sqrt(cantaue)
      MHrate<-sum(dnorm(y,int+cansige*apply(curves,1,sum),cansige,log=T))
      MHrate<-MHrate-sum(dnorm(y,int+sige*apply(curves,1,sum),sige,log=T))
      MHrate<-MHrate+dgamma(cantaue,ae,rate=be,log=T)-dgamma(taue,ae,rate=be,log=T) 
      if(runif(1,0,1)<exp(MHrate)){taue<-cantaue;sige<-cansige}
    }

   #new intercept
    int<-rnorm(1,mean(y-sige*apply(curves,1,sum)),sige/sqrt(n))

   #new curves
   for(j in 1:ncurves){
     ncp<-min(c(median(keepr[,j]),2))
     #first draw the sd:
      rrr<-y-int-sige*apply(curves[,-j],1,sum)
      z<-sqrt(taue)*t(Gamma[j,,])%*%rrr
      for(jjj in 1:npts){
         mxgx[jjj]<-g(grid[jjj],z,sige,D[,j],priorprob=priorprob)/
                    dcan(grid[jjj],ncp,priorprob=priorprob)
      } 
      highpt<-1.05*max(mxgx)
      ratio<-0
      while(ratio<1){
        r[j]<-rcan(1,ncp,priorprob)
        ratio<-g(r[j],z,sige,D[,j],priorprob=priorprob)/highpt/runif(1,0,1)/
               dcan(r[j],ncp,priorprob=priorprob)
       }
  
    #then draw the curve
      if(r[j]==0){curves[,j]<-0}
      if(r[j]>0){
        var<-1/(1+D[,j]/r[j])
        curves[,j]<-Gamma[j,,]%*%(rnorm(nterms,0,sqrt(var))+var*z)
      }
   }

   #Record results:
   keepr[i,]<-r  
   keepl2[i,]<-apply(curves^2,2,mean)
   fit<-int+sige*apply(curves,1,sum)
   dev[i]<- -2*sum(dnorm(y,fit,sige,log=T))
   keepsige[i]<-sige
   keepint[i]<-int
   ks[i]<-sum(r>0)

   mx = apply(x,2,mean)
   my = apply(sige*curves,2,mean)
   Sxx = apply(x^2,2,mean) - mx^2
   Sxy = apply(sige*curves*x,2,mean) - mx*my
   slopes[i,]<-Sxy/Sxx
   intercepts[i,]<-my-(Sxy/Sxx)*mx

   if(i>burn){
      afterburn<-afterburn+1
      sumfit<-sumfit+fit
      sumfit2<-sumfit2+fit^2
      suminout<-suminout+ifelse(r>0,1,0)
      sumcurves<-sumcurves+sige*curves
      sumcurves2<-sumcurves2+(sige*curves)^2
    }
    if(i%%10==0){ print(i) }

    #display current value of the chain
    if(i%%update==0){ 
     par(mfrow=c(2,2))
     if(p>0){plot(x[o1,1],y[o1],main=i);
       lines(x[o1,1],int+sige*curves[o1,1],col=4)}    
     if(p>1){plot(x[o2,2],y[o2],main=i);
       lines(x[o2,2],int+sige*curves[o2,2],col=4)}    
     if(p>2){plot(x[o3,3],y[o3],main=i);
       lines(x[o3,3],int+sige*curves[o3,3],col=4)}    
     if(p>3){plot(x[o4,4],y[o4],main=i);
       lines(x[o4,4],int+sige*curves[o4,4],col=4)}    
    }
  }
  stop<-proc.time()[3]
  print(paste("Sampling took",round(stop-start),"seconds"))

  #Calculate posterior means:
  fitmn<-sumfit/afterburn
  fitsd<-sqrt(sumfit2/afterburn-fitmn^2)
  curves<-sumcurves/afterburn
  curvessd<-sqrt(sumcurves2/afterburn-curves^2)
  probin<-suminout/afterburn
  kpost<-hist(ks[burn:runs]+1,breaks=seq(0,p+1),plot=F)$density


#########  Definitions of the output    ################:
# fittedvalues,fittedsds are the posterior means and sds 
#                        of f at the data points:
# inprob is the posterior inclusion probability
# l2 is the posterior distribution of the l2 norm of each component
# curves and curvessd are the posterior means and sds of the
#                     individual components f_{ij}
# r is the posterior distribution of the variance r
# term1 and term2 index the compnents f_{ij}.  For example, if
#     term1[j]=3 and term2[j]=4 then curves[,j] is the posterior 
#     mean of f_{3,4}.  Terms with terms1[j]=terms2[j] 
#     are main effects
# dev is the posterior samples of the deviance
# int is the posterior samples of the intercept
# sigma is the posterior samples of the error sd



list(fittedvalues=fitmn,fittedsds=fitsd,inprob=probin,l2=keepl2[burn:runs,],
curves=curves,curvessd=curvessd,r=keepr[burn:runs,],
term1=term1,term2=term2,dev=dev,int=keepint,sigma=keepsige,elapsed=stop-start,kposterior=kpost,slope=slopes,intercept=intercepts)}




priorr<-function(r,priorprob=0.5){ifelse(r==0,1-priorprob,priorprob*2*dt(sqrt(r)/lambda,1)/lambda)}

rcan<-function(n,ncp,priorprob=0.5){
  if(priorprob==1){rrr<-abs(rt(n,1,ncp=ncp))}
  if(priorprob<1){rrr<-ifelse(runif(n,0,1)<0.975,abs(rt(n,1,ncp=ncp)),0)}
rrr}

dcan<-function(r,ncp,priorprob=0.5){
  if(priorprob==1){rrr<-abs(rt(n,1,ncp=ncp))}
  if(priorprob<1){rrr<-ifelse(r==0,1-0.975,0.975*2*dt(r,1,ncp=ncp))}
rrr}
g<-function(r,z,sige,d,priorprob=.5){prod(dnorm(z,0,sqrt(1+r/d)))*priorr(r,priorprob)}

#Define Bernoulli polynomials
B0<-function(x){1+0*x}
B1<-function(x){x-.5}
B2<-function(x){x^2-x+1/6}
B3<-function(x){x^3-1.5*x^2+.5*x}
B4<-function(x){x^4-2*x^3+x^2-1/30}
B5<-function(x){x^5-2.5*x^4+1.667*x^3-x/6}
B6<-function(x){x^6-3*x^5+2.5*x^4-.5*x^2+1/42}

makeCOV.ME<-function(xxx,const=10){
  sss<-matrix(xxx,length(xxx),length(xxx),byrow=T)
  ttt<-matrix(xxx,length(xxx),length(xxx),byrow=F)
  diff<-as.matrix(dist(xxx,diag=T,upper=T))

const*(B1(sss)*B1(ttt)+B2(sss)*B2(ttt)/4)-B4(diff)/24}      
    
makeCOV.ME.cat<-function(xxx){
  g<-length(unique(xxx))
  sss<-matrix(xxx,length(xxx),length(xxx),byrow=T)
  ttt<-matrix(xxx,length(xxx),length(xxx),byrow=F)
  equals<-ifelse(sss==ttt,1,0)
(g-1)*equals/g -(1-equals)/g}      
 
makeCOV.INT<-function(xxx1,xxx2,const=10){
  sss1<-matrix(xxx1,length(xxx1),length(xxx1),byrow=T)
  ttt1<-matrix(xxx1,length(xxx1),length(xxx1),byrow=F)
  diff1<-as.matrix(dist(xxx1,diag=T,upper=T))
  KP1<-B1(sss1)*B1(ttt1)+B2(sss1)*B2(ttt1)/4
  KN1<- -B4(diff1)/24      
  if(length(unique(xxx1))<10){KP1<-0*KP1;KN1<-makeCOV.ME.cat(xxx1)}

  sss2<-matrix(xxx2,length(xxx2),length(xxx2),byrow=T)
  ttt2<-matrix(xxx2,length(xxx2),length(xxx2),byrow=F)
  diff2<-as.matrix(dist(xxx2,diag=T,upper=T))
  KP2<-B1(sss2)*B1(ttt2)+B2(sss2)*B2(ttt2)/4
  KN2<- -B4(diff2)/24      
  if(length(unique(xxx2))<10){KP2<-0*KP2;KN2<-makeCOV.ME.cat(xxx2)}

(KP1+KN1)*(KP2+KN2)-(const-1)*(KP1*KP2)}


# _____________________________________________________________________________________________________
# Run on data

# tag determines which example to run
tag<-"bssanova"
#tag<-"gp-bssanova"

#ns<-c(100,500,1000,2000,5000)
ns<-c(100,1000,5000)  # sample sizes to use
runs<-10000  #number of MCMC samples
burn<-1000   #Toss out these many

nns<-length(ns)
totaltime<-rep(0,nns)
samplingtime<-rep(0,nns)
ffile<-F

for(i in 1:nns){
    n<-ns[i]
    infile<-paste0("bssanova-output/",tag,"-data-n=",n,".dat")
    print(infile)
    D <- read.table(infile)
    y <- D[,1]
    x <- D[,2:6]
    n <- nrow(x)  # sample size
    p <- ncol(x)  # number of covariates

    update<-runs+1   #How often to display the current iteration?
    cat<-c(F,F,F,F,F)    #Which variables are categorical?
    n.terms<-100 #n  #Number of eigenvalues to keep
    lambda<-2    #Half-Cauchy hyperparameter            
    interactions<-F  #Include interactions?


    #fit the model:
    start<-proc.time()[3]
    fit<-BSSANOVA(y,x,categorical=cat,
                  lambda=lambda,nterms=n.terms,twoway=interactions,
                  runs=runs,burn=burn,update=update,fromfile=ffile,tag=tag)
    stop<-proc.time()[3]
    print(paste("BSSANOVA took",round(stop-start),"seconds"))
    totaltime[i]<-stop-start
    samplingtime[i]<-fit$elapsed

    #plot the overall function vs x1
    par(mfrow=c(1,1))
    ooo<-order(x[,1])
    g0 <- -1 + 4*x[ooo,1]
    #plot(x[ooo,1],y[ooo]-g0,main="Overall function")
    plot(x[ooo,1],g0-g0,lty=3)
    lines(x[ooo,1],fit$fittedvalues[ooo]-g0)
    lines(x[ooo,1],fit$fittedvalues[ooo]-g0-2*fit$fittedsds[ooo],lty=2)
    lines(x[ooo,1],fit$fittedvalues[ooo]-g0+2*fit$fittedsds[ooo],lty=2)

    #plot the main effect for x1
    par(mfrow=c(1,1))
    ooo<-order(x[,1])
    g0 <- -1 + 4*x[ooo,1]
    plot(x[ooo,1],y[ooo]-g0,main="Main effect 1")
    lines(x[ooo,1],fit$curves[ooo,1]+mean(fit$int)-g0)
    lines(x[ooo,1],fit$curves[ooo,1]+mean(fit$int)-g0-2*fit$curvessd[ooo,1],lty=2)
    lines(x[ooo,1],fit$curves[ooo,1]+mean(fit$int)-g0+2*fit$curvessd[ooo,1],lty=2)

    #print(fit$kposterior)  # posterior on # of vars included
    #print(fit$elapsed)  # time
    #print(fit$int)  # samples of intercept (beta_1)
    #print(fit$slope)  # samples of slopes of best linear fit to each component (beta_2,...,beta_6)
    #print(fit$intercept)  # samples of intercepts of best linear fit to each component (beta_2,...,beta_6)

    # compute beta
    totalint <- (fit$int) + apply(fit$intercept,1,sum)
    beta <- cbind(totalint,fit$slope)

    # save values
    write.table(beta, file=paste0(tag,"-n=",n,"-beta.dat"), sep="\t", row.names=F, col.names=F)
    write.table(fit$kposterior, file=paste0(tag,"-n=",n,"-kposterior.dat"), sep="\t", row.names=F, col.names=F)
    write.table(cbind(fit$fittedvalues,fit$fittedsds), file=paste0(tag,"-n=",n,"-fit.dat"), sep="\t", row.names=F, col.names=F)
    write.table(fit$int, file=paste0(tag,"-n=",n,"-int.dat"), sep="\t", row.names=F, col.names=F)
    write.table(fit$curves, file=paste0(tag,"-n=",n,"-curves.dat"), sep="\t", row.names=F, col.names=F)
    write.table(fit$curvessd, file=paste0(tag,"-n=",n,"-curvessd.dat"), sep="\t", row.names=F, col.names=F)

}
write.table(totaltime, file=paste0(tag,"-totaltime.dat"), sep="\t", row.names=F, col.names=F)
write.table(samplingtime, file=paste0(tag,"-samplingtime.dat"), sep="\t", row.names=F, col.names=F)






























