# A comment: this is a sample script.
#Sys.getenv("R_LIBS_USER")

library("TDA")

Type= commandArgs()


input=paste("/mnt/home/bramerda/Documents/Centrality/365/364_full/",Type[7],".pdb",sep="")
#input="/mnt/home/bramerda/Documents/Persistent_Homology/Test/1yzm.pdb"
#input=paste(Type[7],".pdb",sep="")
full_data=read.fwf(input,widths=c(4,7,4,17,7,8,8,5,10,20))

# subset
mydata=subset(full_data[,c(2,3,5,6,7,9,10)])
mydata=apply(mydata,2,function(x)gsub('\\s+', '',x))
mydata = data.frame(mydata)
rownames(mydata) = 1:nrow(mydata)

colnames(mydata)= c("Index","Type","X", "Y","Z","BF","Atom")
mydata[,c("Index","X", "Y","Z","BF")]=sapply(mydata[,c("Index","X", "Y","Z","BF")],as.factor)

CA = mydata[mydata$Type == "CA"|mydata$Type == Type[6],]
#CA = mydata[mydata$Type == "CA",]


rownames(CA) = 1:nrow(CA)

#print(CA)

C_t = as.matrix(sapply(CA[,c("X","Y","Z")], as.numeric))
#print("Constructing distance matrix...")
m = as.matrix(dist(C_t),method="euclidean")
#r_c = 8 # cutoff

#Con = m
#for (i in 1:nrow(m)){
#  for (j in 1:nrow(m)){
#     if(Con[i,j] > r_c){
#      Con[i,j]=30
#    }
#  }  
#}
#print("Constructing characteristic matrix...")
#characteristic distance matrix M
#print(CA[175:177,])


#mydata["Was_1"]=NA
#CA$Total_Bar = NA

md = 2  #max dimension
ms = 1 #max scale

full_total=0
#print("Running persistence calculation for master complex")
#FullAlphaComplex =alphaComplexDiag(X = full, printProgress = TRUE)
#CX=ripsDiag(X=Con,maxdimension = md, maxscale = ms, dist="arbitrary",printProgress = FALSE)
#for (j in 1:nrow(CX$diagram)){
#  if(CX$diagram[j,1]==1){
#    full_total=full_total+ CX$diagram[j,3]-CX$diagram[j,2]
#  }
#}
#print(tail(CA))

#CA$Full_Bar=full_total  
#cutoff=as.numeric(Type[8])
#print(c('L','CC','P-value'))

result = data.frame(Cutoff=numeric(),
                      P_value=numeric(),
                      CC = numeric(),
                      stringsAsFactors=FALSE
                      )
#eta = 6
#Exponential eta = 7 cutoff = 6 1aba 0.62


#for (cutoff in 3:10){
#lorentzcol  
#cutoff = 7; eta = 21.0;sigma = 5.0 #CC Local_CX_Lor
#cutoff = 7; eta = 2.0;sigma = 1.0 #CC_Lor2 
#exp
cutoff = 9; eta = 10.0;sigma = 1.0 #Local_CX, 0.708
#cutoff = as.numeric(Type[8]); eta = 10.0;sigma = 1.0 #Local_CX, 0.708

#print(cutoff)
#print("Running persistence calculation")
#test = data.frame(matrix(unlist(CX), nrow=nrow(CA), byrow=T),stringsAsFactors=FALSE)
count=0
#stop('okay')

for (i in 1:nrow(CA)){
#for (i in as.integer(Type[8]):1){ 
  #print(rbind(i,nrow(CA)))
  print(i)
  if(CA[i,"Type"]=="CA"){
    loc_filen = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Results/',toString(Type[7]),'/BD_Image/Loc_C',toString(Type[6]),'_', toString(sprintf("%010d",i)),'.csv',sep="")
    glo_filen = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Results/',toString(Type[7]),'/BD_Image/Glo_C',toString(Type[6]),'_', toString(sprintf("%010d",i)),'.csv',sep="")
    if(file.exists(loc_filen) & file.exists(glo_filen)){
      print('file exists')} else {
    local_df = CA[FALSE,]
        for (j in 1:nrow(CA)){
      if(m[i,j]<cutoff&i!=j){
        local_df=rbind(CA[j,],local_df) } }
    global_df = rbind(CA[i,],local_df)
    L_t = as.matrix(sapply(local_df[,c("X","Y","Z")], as.numeric))
    G_t = as.matrix(sapply(global_df[,c("X","Y","Z")], as.numeric))
    
    L_t = as.matrix(dist(L_t),method="euclidean")
    G_t = as.matrix(dist(G_t),method="euclidean")
    
    Loc = matrix(, nrow = nrow(local_df), ncol = nrow(local_df))
    Glo = matrix(, nrow = nrow(global_df), ncol = nrow(global_df))

    for (j in 1:nrow(local_df)){
      k=j
      while (k<=nrow(local_df)){
        #kern  = 1.00/(1.00+(L_t[j,k]/eta)**(sigma))
        kern =  exp(-1.00*((L_t[j,k]/eta))**(sigma))
        Loc[j,k]=kern
        Loc[k,j]=kern
        k=k+1      }
      Loc[j,j]=0      }
    
    for (j in 1:nrow(global_df)){
      k=j
      while (k<=nrow(global_df)){
        #kern  = 1.00/(1.00+(G_t[j,k]/eta)**(sigma))
        kern = exp(-1.00*((G_t[j,k]/eta))**(sigma))
        Glo[j,k]=kern
        Glo[k,j]=kern
        k=k+1    }
      Glo[j,j]=0    }
    
    Local_CX  = ripsDiag(X=Loc,maxdimension = md, maxscale = ms, dist="arbitrary",printProgress = FALSE)
    Global_CX = ripsDiag(X=Glo,maxdimension = md, maxscale = ms, dist="arbitrary",printProgress = FALSE)
    Local = as.data.frame(do.call(rbind,Local_CX))
    Global = as.data.frame(do.call(rbind,Global_CX))
    write.csv(Local,file=loc_filen)
    write.csv(Global,file=glo_filen)
    #write.table(Local, file = loc_filen, sep = "\t",
    #row.names = TRUE, col.names = NA)
    #write.table(Global, file = glo_filen, sep = "\t",
    #row.names = TRUE, col.names = NA)
    }   
  }
}



stop("stopped here")

lmp = function (modelobject) {
    if (class(modelobject) != "lm") stop("Not an object of class 'lm' ")
    f <- summary(modelobject)$fstatistic
    p <- pf(f[1],f[2],f[3],lower.tail=F)
    attributes(p) <- NULL
    return(p)
}

CA = CA[CA$Type == "CA",]


#head(CA)
#tail(CA)
#stop('h9')

#dataset = CA[,c('BF','Was_0','Btl_0')]

#fit = lm(BF ~ Was_0 + Btl_0 + Was_1 + Btl_1, data=CA)
#fit = lm(BF ~ Was_0 + Btl_0, data=CA)
#names(summary(fit))
#print(c(cutoff,sqrt(summary(fit)$r.squared),lmp(fit))) # show results
#result = rbind(c(cutoff,lmp(fit),sqrt(summary(fit)$r.squared)),result)
#}

#print(result)
#filename = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Test/1yzm_C',toString(Type[6]),'_Dist.csv',sep="")
filename = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Results/',toString(Type[7]),'/Local_C',toString(Type[6]), toString(Type[8]),'.csv',sep="")
#filename = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Results/',toString(Type[7]),'/Local_C',toString(Type[6]),'_Lor.csv',sep="")
write.csv(CA,file=filename) #,append=FALSE)
print("Success")
stop("done")

res = paste('/mnt/home/bramerda/Documents/Persistent_Homology/Results/',toString(Type[7]),'/Res_C',toString(Type[6]),'.csv',sep="")
write.csv(result,file=res) #,append=FALSE)

stop("done")

