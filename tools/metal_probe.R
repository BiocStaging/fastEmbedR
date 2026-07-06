library(fastEmbedR)
cat('lib=', find.package('fastEmbedR'), '\n')
cat('metal=', fastEmbedR:::embedding_metal_available_cpp(), '\n')
tryCatch({ fastEmbedR:::standardize_metal_cpp(matrix(rnorm(20),5,4)); cat('std ok\n') }, error=function(e) cat('std err=', conditionMessage(e), '\n'))
