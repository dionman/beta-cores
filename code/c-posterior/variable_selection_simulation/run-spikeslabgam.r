
library(spikeSlabGAM)

# Run on data
ns<-c(100,500,1000,2000,5000)
nns<-length(ns)
totaltime<-rep(0,nns)

for(i in 1:nns){
    n<-ns[i]
    infile<-paste0("bssanova-output/bssanova-data-n=",n,".dat")
    print(infile)
    D <- read.table(infile)
    y <- D[,1]
    x <- D[,2:6]
    colnames(x) <- paste("x", 1:5, sep = "")
    d <- data.frame(y,x)
    f <- y ~ x1+x2+x3+x4+x5
    mcmcset <- list(nChains=1, chainLength=9000, burnin=1000, thin=1, verbose=T, returnSamples=T,
                    sampleY=T, useRandomStart=T, blocksize=50, modeSwitching=0.05, reduceRet=F)

    #fit the model:
    start<-proc.time()[3]
    m <- spikeSlabGAM(formula=f, data=d, mcmc=mcmcset)
    stop<-proc.time()[3]
    print(paste("spikeSlabGAM took",round(stop-start),"seconds"))
    totaltime[i]<-stop-start

}
write.table(totaltime, file="spikeslabgam-totaltime.dat", sep="\t", row.names=F, col.names=F)


