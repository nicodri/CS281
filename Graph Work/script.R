#### Loading the data:
library(igraph)
library(dendextend)

sadj_gs <- read.csv("~/Desktop/R_281/sadj_gs.csv", header=FALSE)
sadj_ovi <- read.csv("~/Desktop/R_281/sadj_ovi.csv", header=FALSE)
sadj_nmf <- read.csv("~/Desktop/R_281/sadj_nfm.csv", header=FALSE)

mat_gs  <- as.matrix(sadj_gs)
mat_ovi  <- as.matrix(sadj_ovi)
mat_nmf  <- as.matrix(sadj_nmf)

cat <- read.csv("~/Desktop/R_281/categories", header=FALSE)

#Generating the graphs:
g_gs <- graph.adjacency(mat_gs,mode = c('undirected'))
g_ovi <- graph.adjacency(mat_ovi,mode = c('undirected'))
g_nmf <- graph.adjacency(mat_nmf,mode = c('undirected'))

#Reducing to connected graphs:
todrop <- c()
for(i in 2:3){
  todrop[i-1] <- (1:1077)[clusters(g_ovi)$membership==i]
}


g_ovi <- delete.vertices(g_ovi,todrop)

todrop <- c()
for(i in 2:9){
todrop[i-1] <- (1:1077)[clusters(g_nmf)$membership==i]
}

g_nmf <- delete_vertices(g_nmf,todrop)


#Naming vertices:
cat_ <- as.matrix(cat)
cat_ <- matrix(cat_, ncol = ncol(cat), dimnames = NULL)

V(g_gs)$name <-  as.vector(cat_)
V(g_ovi)$name <-  as.vector(cat_[-c(todrop1,todrop2)])
V(g_nmf)$name <-  as.vector(cat_[-todrop])

#Fitting the walktraps:
wt_gs <- walktrap.community(g_gs, steps=500,modularity=TRUE)
wt_ovi <- walktrap.community(g_ovi, steps=500,modularity=TRUE)
wt_nmf <- walktrap.community(g_nmf, steps=500,modularity=TRUE)

#Creating the dendograms:
den_gs <- as.dendrogram(wt_gs, use.modularity=TRUE)
den_ovi <- as.dendrogram(wt_ovi, use.modularity=TRUE)
den_nmf <- as.dendrogram(wt_nmf,use.modularity = TRUE)

#Coloring the edges by categories:
colorCodes <- c(Seafood="red", Mexican="green", Chinese="blue", Sports ="yellow", Sushi="pink", Steakhouses="black", B_B="purple")
labels_colors(den_gs) <- colorCodes[as.vector(cat_)][order.dendrogram(den_gs)]
labels_colors(den_ovi) <- colorCodes[as.vector(cat_[-c(todrop1,todrop2)])][order.dendrogram(den_ovi)]
labels_colors(den_nmf) <- colorCodes[as.vector(cat_[-todrop])][order.dendrogram(den_nmf)]


#Plotting:
pdf("~/Desktop/R_281/den_gs.pdf", width=70, height=15)
plot(den_gs)
dev.off()

pdf("~/Desktop/R_281/den_ovi.pdf", width=70, height=15)
plot(den_ovi)
dev.off()

pdf("~/Desktop/R_281/den_nmf.pdf", width=70, height=15)
plot(den_nmf)
dev.off()
