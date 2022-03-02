##### Script for sampling from the posterior predictive of each experiment in [1], as processed by bayesynergy
# 
# Input: 
#         Individual_Experiments/raw/cell_line : drug_A + drug_B.RData
#         .RData file containing each combination experiment fit
#         Individual_Experiments/mono/raw/cell_line : drug_A.RData
#         .RData file containing single monotherapy fit for drug A
#         Individual_Experiments/mono/raw/cell_line : drug_B.RData
#         .RData file containing single monotherapy fit for drug B

# Output:
#         Individual_Experiments/raw/cell_line : drug_A + drug_B.RData
#         .RData file containing each single experiment fitted with bayesynergy

dir.create(file.path("Individual_Experiments/postPred/"), showWarnings = FALSE)

rm(list=ls())

library(foreach)
library(stringr)

# Set up cluster
#cl = parallel::makeCluster(50,outfile="debug_postProcess.txt",type="FORK")
#doParallel::registerDoParallel(cl=cl)
#foreach::getDoParRegistered()

files = list.files("Individual_experiments/raw",pattern="*.RData")

print(paste0("Files left to process: ",length(files)))

#foreach(zzz = 1:length(files)) %dopar% {
for (zzz in 1:length(files)){
  ####### THE COMBINATION
  load(paste0("Individual_experiments/raw/",files[zzz]))
  # First, set up grid to predict over
  x1 = sort(log10(unique(fit$data$x[,1])))[-1]
  x2 = sort(log10(unique(fit$data$x[,2])))[-1]
  
  # Scale to [0,1]
  z1 = (x1-min(x1))/(max(x1)-min(x1))
  z2 = (x2-min(x2))/(max(x2)-min(x2))
  
  # Grid to predict on at scaled values
  n.pred = 10
  Z1 = seq(0,1,length.out=n.pred)
  Z2 = seq(0,1,length.out=n.pred)
  # Scaled up to original scale
  X1 = Z1*(max(x1)-min(x1))+min(x1)
  X2 = Z2*(max(x2)-min(x2))+min(x2)
  
  # Create distance matrices needed for calculations
  x1_train_dist = abs(outer(x1,x1,FUN="-"))
  x2_train_dist = abs(outer(x2,x2,FUN="-"))
  X1_test_dist = abs(outer(X1,X1,FUN="-"))
  X2_test_dist = abs(outer(X2,X2,FUN="-"))
  # Cross distances
  X1x1_cross_dist = abs(outer(X1,x1,FUN="-"))
  X2x2_cross_dist = abs(outer(X2,x2,FUN="-"))
  
  ### Function for kronecker products and reshaping
  kron_prod = function(A,B,V){
    # returns (A \otimes B)*vec(V)
    return (t(A%*%t(B%*%V)))
  }
  kron_mmprod = function(A,B,M){
    # extends kron_prod to matrix multiplication
    res = c()
    for (i in 1:ncol(M)){
      res = c(res,as.vector(kron_prod(A,B,matrix(M[,i],ncol=sqrt(nrow(M))))))
    }
    return (matrix(res,ncol=ncol(M)))
  }
  
  
  # Pull out posterior
  post = rstan::extract(fit$stanfit)
  n.samples = length(post$lp__)
  # Set up container to store predictions
  GP.hat = array(NA,dim=c(n.samples,n.pred,n.pred))
  p0.hat = array(NA,dim=c(n.samples,n.pred,n.pred))
  f.hat = array(NA,dim=c(n.samples,n.pred,n.pred))
  
  for (i in 1:n.samples){
    # For current parameters, evaluate latent GP at new locations
    sigma_f = post$sigma_f[i]
    ell = post$ell[i]
    poly1 = (1+sqrt(3)*(x1_train_dist / ell))
    poly2 = (1+sqrt(3)*(x2_train_dist / ell))
    poly1_xx = (1+sqrt(3)*(X1_test_dist / ell))
    poly2_xx = (1+sqrt(3)*(X2_test_dist / ell))
    cov1 = sigma_f*(poly1*exp(-sqrt(3)*x1_train_dist / ell))
    cov2 = poly2*exp(-sqrt(3)*x2_train_dist / ell)
    cross_poly1 = (1+sqrt(3)*(X1x1_cross_dist / ell))
    cross_poly2 = (1+sqrt(3)*(X2x2_cross_dist / ell));
    cross_cov1 = sigma_f*(cross_poly1*exp(-sqrt(3)*X1x1_cross_dist / ell))
    cross_cov2 = cross_poly2*exp(-sqrt(3)*X2x2_cross_dist / ell)
    cov1_xx = sigma_f*(poly1_xx*exp(-sqrt(3)*X1_test_dist / ell))
    cov2_xx = poly2_xx*exp(-sqrt(3)*X2_test_dist / ell)
    
    # Need to invert cov1 and cov2
    L1 = solve(cov1 + 1e-8*diag(nrow(cov1)))
    L2 = solve(cov2 + 1e-8*diag(nrow(cov2)))
    y = post$GP[i,,]
    GPmean = kron_prod(cross_cov1,cross_cov2,kron_prod(L1,L2,y))
    GPvar = kronecker(cov1_xx,cov2_xx) - kron_mmprod(cross_cov1,cross_cov2,kron_mmprod(L1,L2,t(kronecker(cross_cov1,cross_cov2))))
    GP = matrix(as.vector(GPmean) + t(chol(GPvar))%*%rnorm(length(as.vector(GPmean))),n.pred)
    GP.hat[i,,] = GP
    
    # For current parameters evaluate p0 at new locations
    la_1 = post$la_1[i]
    la_2 = post$la_2[i]
    log10_ec50_1 = post$log10_ec50_1[i]
    log10_ec50_2 = post$log10_ec50_2[i]
    slope_1 = post$slope_1[i]
    slope_2 = post$slope_2[i]
    P01 = la_1+(1-la_1)/(1+10^(slope_1*(X1-log10_ec50_1)))
    P02 = la_2+(1-la_2)/(1+10^(slope_2*(X2-log10_ec50_2)))
    P0 = outer(P02,P01)
    p0.hat[i,,] = P0
    
    # For current parameters evaluate f at new locations
    # First obtain Delta
    b1 = post$b1[i]
    b2 = post$b2[i]
    Delta = -P0/(1+exp(b1*GP+log(P0/(1-P0))))+(1-P0)/(1+exp(-b2*GP-log(P0/(1-P0))))
    # Then get f
    f.hat[i,,] = P0 + Delta
  }
  
  f.mean = apply(f.hat,c(2,3),mean)
  f.Var = apply(f.hat,c(2,3),var)
  GPMean = apply(GP.hat,c(2,3),mean)
  GPVar = apply(GP.hat,c(2,3),var)
  b1 = mean(post$b1)
  b2 = mean(post$b2)
  
  ### THE MONOTHERAPIES
  # Pull out cell line and drug names
  cell = str_trim(str_split(files[zzz],":")[[1]][1])
  
  ## Get drug names
  drugA = str_trim(str_split(str_split(files[zzz],":")[[1]][2],"&")[[1]][1])
  drugB = str_trim(str_split(str_split(str_split(files[zzz],":")[[1]][2],"&")[[1]][2],".RData")[[1]][1])
  # Find corresponding monotherapy experiments
  # MONO_A
  load(paste0("Individual_Experiments/mono/raw/",cell," : ",drugA,".RData"))
  postA = rstan::extract(fit)
  # MONO_B
  load(paste0("p0/results/raw/",cell," : ",drugB,".RData"))
  postB = rstan::extract(fit)
  
  n.samples = length(postA$lp__)
  # Set up container to store predictions
  p0.hat = array(NA,dim=c(n.samples,n.pred,n.pred))
  for (i in 1:n.samples){
    
    # For current parameters evaluate p0 at new locations
    la_1 = postA$la_1[i]
    la_2 = postB$la_1[i]
    log10_ec50_1 = postA$log10_ec50_1[i]
    log10_ec50_2 = postB$log10_ec50_1[i]
    slope_1 = postA$slope_1[i]
    slope_2 = postB$slope_1[i]
    P01 = la_1+(1-la_1)/(1+10^(slope_1*(X1-log10_ec50_1)))
    P02 = la_2+(1-la_2)/(1+10^(slope_2*(X2-log10_ec50_2)))
    P0 = outer(P02,P01)
    p0.hat[i,,] = P0
    
  }
  
  p0.mean = apply(p0.hat,c(2,3),mean)
  
  
  df = data.frame("CellLine" = cell,"Drug A" = drugA,
                  "Drug B" = drugB,
                  "Drug A conc" = sort(rep(Z1,length(Z2))),
                  "Drug B conc" = rep(Z2,length(Z1)),
                  "GPMean" = as.vector(GPMean), "GPVar" = as.vector(GPVar),
                  "fMean" = as.vector(f.mean), "fVAR" = as.vector(f.Var),
                  "p0Mean" = as.vector(p0.mean),
                  "b1" = b1, "b2" = b2)
  
  # Write df table to .csv file
  write.table(df,file=paste0("Individual_Experiments/postPred/",sub("\\.RData","",files[zzz]),".csv"),sep=";",row.names = F)
}

  
  
    


