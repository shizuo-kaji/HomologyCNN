#######################################
#
# Persistent homology of images
#                by S. Kaji
#                Oct. 2017
#
# Requires: ggplot2, TDA, imager packages
# install them using the following lines if they do not exist in your system
# install.packages(c("ggplot2","TDA","imager"))
#######################################

## parameters (global variable)
min_length <- 5      # the minimum length of 1-cycles to be displayed
max_length <- 5000    #  clip the length(color) of 1-cycles 
min_life <- 0.1       # the minimum life-time of cycles to be displayed
max_life <- 1.0        # the maximum life-time of cycles to be displayed
sub=F                 # set FALSE for superlevel filtration 
verbose=F         # print information
normalise=F       # normalise images before PH computation
lb <- 0        # only pixels with lb < y < ub are used for PD computation
ub <- 3000

###############################################################
library(TDA)
library(imager)
library(ggplot2)
#library(pforeach)

## compute PD of a function over the grid
PD <- function(X, progress=F, sub=FALSE, loc=FALSE){
  d <- dim(X)
  Xlim <- c(1, d[1]);  Ylim <- c(1, d[2]);  by <- 1
  Diag <- gridDiag(FUNvalues = X, lim = cbind(Xlim, Ylim),
                   by = by, sublevel = sub, library = "Dionysus",
                   printProgress = progress, location =loc)
  return(Diag)
}

# plot persistence diagram
plotPD <- function(D,name="Noname"){
  c <- as.factor(D[,1])
  xylim <- c(min(D[,2:3]),max(D[,2:3]))
  q <- qplot(x=(D[,2]),y=(D[,3]), col=c, main=name, 
             xlim=xylim, ylim=xylim)
  print(q)
  return(q)
}

# write the list of image files in a text file
writeList <- function(imgfiles,txtfile){
  out <- file(txtfile, "w")
  for(f in imgfiles){
    writeLines(paste(f), out, sep="\t")
    #  writeLines(paste("0\t0"), out, sep="\n")
    label <- substr(f,1,1)
    writeLines(paste(label), out, sep="\n")
  }
  close(out) 
}

## weighted betti
wbetti <- function(f,d){
  ims <- grayscale(load.image(f))
  #  ims <- t(as.matrix(read.csv(f, sep=",", header=F)))
  #  image(ims)
  Diag <- PD(ims[,], progress=verbose, sub=sub, loc=F)
  D <- Diag$diagram
  D <- D[D[,1]==d,]
  return(sum(sqrt(abs(D[,2]-D[,3]))))
}

## highlight cycle generators
highlight_cycles <- function(diag,cycleidx,img){
  D <- diag[["diagram"]]
  DL <- diag[["cycleLocation"]]
  plot(img, axes=F)#,rescale=F)
  palette(heat.colors(100))
  #highlight(px.circle(r = 10, x = 600, y = 676), col = "orange")
  for (i in seq(along = cycleidx)) {
    life <- abs(D[cycleidx[i],3]-D[cycleidx[i],2])
    for (j in seq_len(dim(DL[[cycleidx[i]]])[1])) {
      if(D[cycleidx[i],1]==0){
        xy <- DL[[cycleidx[i]]][j, , ]
        points(xy[1],xy[2],pch=1,col=ceiling(life*500))
      }else{
        lines(DL[[cycleidx[i]]][j, , ], pch = 19, cex = 1, col = ceiling(life*500))
      }
    }
  }
}

## computation of PH images
PHimg <- function(f,d){
  if(fileext=="csv"){
    ims <- t(as.matrix(read.csv(f, sep=",", header=F)))
    subims <- ims[lb:min(ub,nrow(ims)),]
  }else{
    ims <- grayscale(load.image(f))
    ## do some preprocessing
    #  ims <- enorm(imgradient(ims,'xy'))
    #  ims <- imhessian(ims,"xy")
    if(normalise){
      ims <- (ims-mean(ims))/sd(ims)
    }
    subims <- imsub(ims, lb < y & y < ub)
  }
  #  plot(as.cimg(subims))
  Diag <- PD(subims[,], progress=verbose, sub=sub, loc=T)
  D <- Diag[["diagram"]]
  DL <- Diag[["cycleLocation"]]
  if(verbose){
    plotPD(D,f)
  }
  if(d==1){
    ## compute H_1 image
    cyclen <- do.call("rbind",lapply(DL,dim))[,1]   # length of cycles
    print(range(cyclen))
    cyclen <- pmax( 0, pmin( cyclen, max_length))   # clip
    cycleidx <- which(D[,1]==1 & abs(D[,3]-D[,2])>min_life & cyclen>min_length & cyclen <= max_length)
    label_im <- array(0,dim(ims)[1:2])
    label_im_len <- array(0,dim(ims)[1:2])
    for (i in seq(along = cycleidx)) {
      life <- abs(D[cycleidx[i],3]-D[cycleidx[i],2])
      life <- pmax( min_life, pmin( life, max_life))
      for (j in seq_len(dim(DL[[cycleidx[i]]])[1])) {
        xy <- DL[[cycleidx[i]]][j, , ]
        ## colour with lifetime
        label_im[xy[1,1],xy[1,2]+lb] <- max(sqrt(life/max_life),label_im[xy[1,1],xy[1,2]+lb])
        label_im[xy[2,1],xy[2,2]+lb] <- max(sqrt(life/max_life),label_im[xy[2,1],xy[2,2]+lb])
        ## colour with cycle length
        label_im_len[xy[1,1],xy[1,2]+lb] <- max(sqrt(cyclen[cycleidx[i]] / max_length),label_im_len[xy[1,1],xy[1,2]+lb])
        label_im_len[xy[2,1],xy[2,2]+lb] <- max(sqrt(cyclen[cycleidx[i]] / max_length),label_im_len[xy[2,1],xy[2,2]+lb])
      }
    }
    #  plot(as.cimg(label_im))
    prefix <- ifelse(sub,"Hsub","Hsup")
    fname <- substr(basename(f), 1, nchar(basename(f)) - 4)
    save.image(as.cimg(label_im),paste0(fname,"_",prefix,"1_life.png"))
    save.image(as.cimg(label_im_len),paste0(fname,"_",prefix,"1_len.png"))
  }else if(d==0){
    ## compute H_0 image
    cycleidx <- which(D[,1]==0 & abs(D[,3]-D[,2])>min_life)
    label_im <- array(0,dim(ims)[1:2])
    for (i in seq(along = cycleidx)) {
      life <- abs(D[cycleidx[i],3]-D[cycleidx[i],2])
      for (j in seq_len(dim(DL[[cycleidx[i]]])[1])) {
        xy <- DL[[cycleidx[i]]][j, , ]
        label_im[xy[1],xy[2]+lb] <- max(label_im[xy[1],xy[2]+lb],sqrt(life))
      }
    }
    save.image(as.cimg(label_im),paste0(fname,"_",prefix,"0.png"))
  }
}
############## End of function definition ####################

#### set the paths to the directory containing image files
args = commandArgs(trailingOnly=TRUE)
path = args[1]     ## dir containing images
fileext <- args[2]  ## image file extention
setwd(path)

imgfiles <- list.files(path, pattern=paste0(".",fileext,"$"))

## batch compute H_1
for(f in imgfiles){
  print(f)
  PHimg(f,d=1)
}


