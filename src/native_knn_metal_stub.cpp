#include <Rcpp.h>

#include <string>

bool native_metal_knn_available_impl() {
  return false;
}

Rcpp::List native_metal_knn_impl(SEXP,
                                 int,
                                 const std::string&,
                                 const std::string&,
                                 double) {
  Rcpp::stop("Native Metal KNN is available only on a Metal-enabled macOS build.");
}
