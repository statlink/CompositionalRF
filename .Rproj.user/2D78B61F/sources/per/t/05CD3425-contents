// Based on the code of MultivariateRandomForest

// [[Rcpp::depends(RcppArmadillo)]]
// #include <RcppArmadillo.h>
#include <Rcpp.h>
#include <RcppParallel.h>
#include <memory>
#include <algorithm>
#include <vector>
#include <cfloat>
#include <cmath>
#include <numeric>
#ifdef RCPP_PARALLEL_USE_TBB
#include <tbb/global_control.h>  // For controlling the number of threads
#endif

using namespace Rcpp;
using namespace RcppParallel;
using std::vector;

// Define a RESTRICT macro for portability.
#if defined(__GNUG__) || defined(__clang__)
#define RESTRICT __restrict__
#else
#define RESTRICT __restrict
#endif

List splitt22(NumericMatrix X, NumericMatrix Y, int m_feature,
              NumericVector Index, NumericMatrix Inv_Cov_Y, int Command,
              NumericVector ff) {
  int n = Index.size();
  
  // Convert 1-indexed Index to 0-indexing for C++
  std::vector<int> indices(n);
  for (int i = 0; i < n; i++) {
    indices[i] = Index[i] - 1;
  }
  
  // Extract subset of X and Y
  int ncolX = X.ncol(), ncolY = Y.ncol();
  NumericMatrix x(n, ncolX), y(n, ncolY);
  for (int j = 0; j < ncolX; j++) {
    for (int i = 0; i < n; i++) {
      x(i, j) = X(indices[i], j);
    }
  }
  for (int j = 0; j < ncolY; j++) {
    for (int i = 0; i < n; i++) {
      y(i, j) = Y(indices[i], j);
    }
  }
  
  // Convert feature indices to 0-based
  std::vector<int> feats(m_feature);
  for (int i = 0; i < m_feature; i++) {
    feats[i] = ff[i] - 1;
  }
  
  double min_score = std::numeric_limits<double>::infinity();
  double best_threshold = 0.0;
  int best_feature = -1;
  std::vector<int> best_left, best_right;
  
  for (int mm = 0; mm < m_feature; mm++) {
    int col = feats[mm];
    
    // Efficient sorting with indexed ordering
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return x(a, col) < x(b, col);
    });
    
    std::vector<double> cum_sum(n, 0.0), cum_sum_sq(n, 0.0);
    if (Command == 1 && ncolY == 1) {
      cum_sum[0] = y(order[0], 0);
      cum_sum_sq[0] = cum_sum[0] * cum_sum[0];
      for (int i = 1; i < n; i++) {
        cum_sum[i] = cum_sum[i - 1] + y(order[i], 0);
        cum_sum_sq[i] = cum_sum_sq[i - 1] + y(order[i], 0) * y(order[i], 0);
      }
    }
    
    for (int k = 0; k < n - 1; k++) {
      int nleft = k + 1, nright = n - nleft;
      
      double left_cost = 0.0, right_cost = 0.0;
      if (Command == 1 && ncolY == 1) {
        left_cost = cum_sum_sq[k] - (cum_sum[k] * cum_sum[k]) / nleft;
        right_cost = (cum_sum_sq[n - 1] - cum_sum_sq[k]) -
          ((cum_sum[n - 1] - cum_sum[k]) * (cum_sum[n - 1] - cum_sum[k])) / nright;
      } else if (Command == 2) {
        std::vector<double> mean_left(ncolY, 0.0), mean_right(ncolY, 0.0);
        for (int j = 0; j < ncolY; j++) {
          for (int i = 0; i < nleft; i++) mean_left[j] += y(order[i], j);
          for (int i = nleft; i < n; i++) mean_right[j] += y(order[i], j);
          mean_left[j] /= nleft;
          mean_right[j] /= nright;
        }
        
        for (int i = 0; i < nleft; i++) {
          for (int j = 0; j < ncolY; j++) {
            double diff = y(order[i], j) - mean_left[j];
            for (int l = 0; l < ncolY; l++) {
              left_cost += diff * (y(order[i], l) - mean_left[l]) * Inv_Cov_Y(j, l);
            }
          }
        }
        
        for (int i = nleft; i < n; i++) {
          for (int j = 0; j < ncolY; j++) {
            double diff = y(order[i], j) - mean_right[j];
            for (int l = 0; l < ncolY; l++) {
              right_cost += diff * (y(order[i], l) - mean_right[l]) * Inv_Cov_Y(j, l);
            }
          }
        }
      }
      
      double total_cost = left_cost + right_cost;
      if (total_cost < min_score) {
        min_score = total_cost;
        best_feature = col + 1;
        best_threshold = (x(order[k], col) + x(order[k + 1], col)) * 0.5;
        
        best_left.assign(order.begin(), order.begin() + nleft);
        best_right.assign(order.begin() + nleft, order.end());
        
        for (int &idx : best_left) idx = indices[idx] + 1;
        for (int &idx : best_right) idx = indices[idx] + 1;
      }
    }
  }
  
  return List::create(
    Named("Idx_left") = best_left,
    Named("Idx_right") = best_right,
    Named("Feature_number") = best_feature,
    Named("Threshold_value") = best_threshold
  );
}


// ----------------------------------------------------------------
// Structure to hold the best split result.
struct SplitResult {
  double min_score;
  double best_threshold;
  int best_feature;
  std::vector<int> best_left;   // R indices (1-indexed)
  std::vector<int> best_right;  // R indices (1-indexed)
  
  SplitResult() 
    : min_score(std::numeric_limits<double>::infinity()),
      best_threshold(0.0),
      best_feature(-1) {}
};

// ----------------------------------------------------------------
// Worker struct to evaluate candidate features in parallel.
// This version builds the ordering in each candidate exactly as in splitt22_same.
struct SplitWorker : public Worker {
  // Dimensions and settings.
  const int n;       // number of rows (selected indices)
  const int ncolX;   // number of columns in x (all features)
  const int ncolY;   // number of columns in y (response)
  const int command; // 1 for univariate, 2 for multivariate/quadratic cost
  
  // Data matrices (wrapped for parallel read-only access).
  const RMatrix<double> x;       // submatrix x [n x ncolX]
  const RMatrix<double> y;       // submatrix y [n x ncolY]
  const RMatrix<double> invCovY; // for command==2
  
  // Vectors for original row indices (from subsetting) and candidate feature columns.
  const std::vector<int>& indices; // original rows (0-indexed)
  const std::vector<int>& feats;   // candidate feature column indices (0-indexed)
  
  // Best result found by this worker.
  SplitResult best;
  
  // Primary constructor.
  SplitWorker(int n, int ncolX, int ncolY, int command,
              const RMatrix<double>& x,
              const RMatrix<double>& y,
              const std::vector<int>& indices,
              const std::vector<int>& feats,
              const RMatrix<double>& invCovY)
    : n(n), ncolX(ncolX), ncolY(ncolY), command(command),
      x(x), y(y), invCovY(invCovY),
      indices(indices), feats(feats), best() { }
  
  // Copy constructor for parallelReduce.
  SplitWorker(const SplitWorker &other)
    : n(other.n), ncolX(other.ncolX), ncolY(other.ncolY),
      command(other.command), x(other.x), y(other.y), invCovY(other.invCovY),
      indices(other.indices), feats(other.feats), best(other.best) { }
  
  // Splitting constructor required by RcppParallel.
  SplitWorker(const SplitWorker &other, Split)
    : n(other.n), ncolX(other.ncolX), ncolY(other.ncolY),
      command(other.command), x(other.x), y(other.y), invCovY(other.invCovY),
      indices(other.indices), feats(other.feats), best() { }
  
  // Operator(): Process candidate features in the range [begin, end)
  void operator()(std::size_t begin, std::size_t end) {
    // We'll build the ordering for each candidate feature as in splitt22_same.
    for (std::size_t mm = begin; mm < end; mm++) {
      int col = feats[mm];
      
      // Build two vectors for the candidate column.
      std::vector<double> xj(n), xjj(n);
      for (int i = 0; i < n; i++) {
        double val = x(i, col);
        xj[i] = val;
        xjj[i] = val;
      }
      
      // Sort xj in increasing order.
      std::sort(xj.begin(), xj.end());
      
      // Build the ordering by matching sorted xj to xjj.
      // This mimics splitt22_same's double-loop with tie handling.
      std::vector<int> order_local(n, -1);
      for (int ii = 0; ii < n; ii++) {
        for (int jj = 0; jj < n; jj++) {
          if (std::fabs(xj[ii] - xjj[jj]) < 1e-12) {
            order_local[ii] = jj; 
            // Mark this entry as used.
            xjj[jj] = -std::numeric_limits<double>::infinity();
            break;
          }
        }
      }
      
      // --- Branch for univariate response:
      if (command == 1 && ncolY == 1) {
        std::vector<double> cum_sum(n), cum_sum_sq(n);
        double y_val = y(order_local[0], 0);
        cum_sum[0] = y_val;
        cum_sum_sq[0] = y_val * y_val;
        for (int i = 1; i < n; i++) {
          double val = y(order_local[i], 0);
          cum_sum[i] = cum_sum[i - 1] + val;
          cum_sum_sq[i] = cum_sum_sq[i - 1] + val * val;
        }
        double total_sum = cum_sum[n - 1];
        double total_sum_sq = cum_sum_sq[n - 1];
        
        // Evaluate every possible split.
        for (int k = 0; k < n - 1; k++) {
          int nleft = k + 1;
          int nright = n - nleft;
          double left_cost = cum_sum_sq[k] - (cum_sum[k] * cum_sum[k]) / static_cast<double>(nleft);
          double right_cost = (total_sum_sq - cum_sum_sq[k]) -
            ((total_sum - cum_sum[k]) * (total_sum - cum_sum[k])) / static_cast<double>(nright);
          double total_cost = left_cost + right_cost;
          if (total_cost < best.min_score) {
            best.min_score = total_cost;
            best.best_feature = col + 1;  // Convert to 1-indexing.
            best.best_threshold = ( x(order_local[k], col) + x(order_local[k + 1], col) ) * 0.5;
            best.best_left.resize(nleft);
            best.best_right.resize(nright);
            for (int i = 0; i < nleft; i++) {
              best.best_left[i] = indices[order_local[i]] + 1;
            }
            for (int i = nleft; i < n; i++) {
              best.best_right[i - nleft] = indices[order_local[i]] + 1;
            }
          }
        }
      } 
      // --- Branch for multivariate / quadratic cost.
      else if (command == 2) {
        int sizeQ = ncolY * ncolY; // flattened outer-product matrix size
        
        // Allocate work arrays: cumulative sums for responses and cumulative outer products.
        std::vector<double> cumL(n * ncolY, 0.0);
        std::vector<double> cumQ(n * sizeQ, 0.0);
        
        // For the first element (k = 0):
        for (int j = 0; j < ncolY; j++) {
          cumL[j] = y(order_local[0], j);
        }
        for (int j = 0; j < ncolY; j++) {
          for (int l = 0; l < ncolY; l++) {
            cumQ[j * ncolY + l] = y(order_local[0], j) * y(order_local[0], l);
          }
        }
        // For k = 1 to n - 1.
        for (int k = 1; k < n; k++) {
          int offsetL = k * ncolY;
          int prevOffsetL = (k - 1) * ncolY;
          for (int j = 0; j < ncolY; j++) {
            cumL[offsetL + j] = cumL[prevOffsetL + j] + y(order_local[k], j);
          }
          int offsetQ = k * sizeQ;
          int prevOffsetQ = (k - 1) * sizeQ;
          for (int j = 0; j < ncolY; j++) {
            int j_ncolY = j * ncolY;
            for (int l = 0; l < ncolY; l++) {
              cumQ[offsetQ + j_ncolY + l] = cumQ[prevOffsetQ + j_ncolY + l] +
                y(order_local[k], j) * y(order_local[k], l);
            }
          }
        }
        
        // Compute total sums for the right partition.
        std::vector<double> totalL(ncolY), totalQ(sizeQ);
        int totalOffsetL = (n - 1) * ncolY;
        int totalOffsetQ = (n - 1) * sizeQ;
        for (int j = 0; j < ncolY; j++) {
          totalL[j] = cumL[totalOffsetL + j];
        }
        for (int idx = 0; idx < sizeQ; idx++) {
          totalQ[idx] = cumQ[totalOffsetQ + idx];
        }
        
        // Evaluate every possible split point.
        for (int k = 0; k < n - 1; k++) {
          int nleft = k + 1;
          int nright = n - nleft;
          double cost_left = 0.0, cost_right = 0.0;
          int offsetQ_k = k * sizeQ;
          int offsetL_k = k * ncolY;
          // Left partition cost.
          for (int j = 0; j < ncolY; j++) {
            int offsetQ_k_L_j = offsetQ_k + j * ncolY;
            int offsetL_k_j = offsetL_k + j;
            for (int l = 0; l < ncolY; l++) {
              double diff = cumQ[offsetQ_k_L_j + l] -
                (cumL[offsetL_k_j] * cumL[offsetL_k + l]) / static_cast<double>(nleft);
              cost_left += diff * invCovY(j, l);
            }
          }
          // Right partition cost.
          for (int j = 0; j < ncolY; j++) {
            int offsetQ_k_L_j = offsetQ_k + j * ncolY;
            int offsetL_k_j = offsetL_k + j;
            for (int l = 0; l < ncolY; l++) {
              double right_sum_j = totalL[j] - cumL[offsetL_k_j];
              double right_sum_l = totalL[l] - cumL[offsetL_k + l];
              double diff = (totalQ[j * ncolY + l] - cumQ[offsetQ_k_L_j + l]) -
                (right_sum_j * right_sum_l) / static_cast<double>(nright);
              cost_right += diff * invCovY(j, l);
            }
          }
          double total_cost = cost_left + cost_right;
          if (total_cost < best.min_score) {
            best.min_score = total_cost;
            best.best_feature = col + 1;
            best.best_threshold = ( x(order_local[k], col) + x(order_local[k + 1], col) ) * 0.5;
            best.best_left.resize(nleft);
            best.best_right.resize(nright);
            for (int i = 0; i < nleft; i++) {
              best.best_left[i] = indices[order_local[i]] + 1;
            }
            for (int i = nleft; i < n; i++) {
              best.best_right[i - nleft] = indices[order_local[i]] + 1;
            }
          }
        }
      }
    } // end for each candidate mm
  }
  
  // join(): Combine results from two workers.
  void join(const SplitWorker &other) {
    if (other.best.min_score < best.min_score) {
      best = other.best;
    }
  }
};

// ----------------------------------------------------------------

List splitt22_parallel(NumericMatrix X, NumericMatrix Y, int m_feature,
                       NumericVector Index, NumericMatrix Inv_Cov_Y,
                       int Command, NumericVector ff, int nCores) {
  // 1. Adjust indices from R's 1-indexing to C++'s 0-indexing.
  int n = Index.size();
  std::vector<int> indices(n);
  for (int i = 0; i < n; i++) {
    indices[i] = Index[i] - 1;
  }
  
  // 2. Build submatrices xSubset and ySubset using only the selected rows.
  int ncolX = X.ncol();
  NumericMatrix xSubset(n, ncolX);
  int X_nrow = X.nrow();
  for (int j = 0; j < ncolX; j++) {
    for (int i = 0; i < n; i++) {
      xSubset[i + j * n] = X[indices[i] + j * X_nrow];
    }
  }
  
  int ncolY = Y.ncol();
  NumericMatrix ySubset(n, ncolY);
  int Y_nrow = Y.nrow();
  for (int j = 0; j < ncolY; j++) {
    for (int i = 0; i < n; i++) {
      ySubset[i + j * n] = Y[indices[i] + j * Y_nrow];
    }
  }
  
  // 3. Adjust candidate feature indices from R's 1-indexing to C++'s 0-indexing.
  std::vector<int> feats(m_feature);
  for (int i = 0; i < m_feature; i++) {
    feats[i] = ff[i] - 1;
  }
  
  // 4. Wrap the submatrices for safe read-only parallel access.
  RMatrix<double> rX(xSubset);
  RMatrix<double> rY(ySubset);
  RMatrix<double> rInvCovY(Inv_Cov_Y);
  
  // 5. Set up and launch the parallel worker.
  SplitWorker worker(n, ncolX, ncolY, Command, rX, rY, indices, feats, rInvCovY);
  
  // Control the maximum number of threads.
  tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nCores);
  parallelReduce(0, static_cast<size_t>(m_feature), worker);
  
  // 6. Return the best split result as an R list.
  return List::create(
    Named("Idx_left") = worker.best.best_left,
    Named("Idx_right") = worker.best.best_right,
    Named("Feature_number") = worker.best.best_feature,
    Named("Threshold_value") = worker.best.best_threshold
  );
}

List splitt2_fun(NumericMatrix X, NumericMatrix Y, int m_feature, 
                 NumericVector Index, NumericMatrix Inv_Cov_Y, 
                 int Command, NumericVector ff, int nCores = 0) {
  
#if RCPP_PARALLEL_USE_TBB
  if (nCores > 0){
    return splitt22_parallel(X, Y, m_feature, Index, Inv_Cov_Y, Command, ff,
                             nCores);
  } else {
    return splitt22(X, Y, m_feature, Index, Inv_Cov_Y, Command, ff);
  }
#else
  return splitt22(X, Y, m_feature, Index, Inv_Cov_Y, Command, ff);
#endif
}

// Structure to hold a node in the tree.
struct TreeNode {
  int nodeid;
  NumericVector indices;  // Using NumericVector to match splitt2_fun
  
  TreeNode(int id, const NumericVector & idx) : nodeid(id), indices(idx) {}
};


List split_node_iterative_cpp(const NumericMatrix &X, 
                              const NumericMatrix &Y,
                              int m_feature, 
                              NumericVector Index,
                              int min_leaf, 
                              const NumericMatrix &Inv_Cov_Y,
                              int Command,
                              int nCores = 0) {
  
  // The "model" is stored as a vector, where the element at index (nodeid-1)
  // corresponds to the node's information.
  vector<RObject> model;
  
  // Utility lambda: set model[nodeid-1] to a value, resizing if needed.
  auto setModel = [&model](int nodeid, RObject value) {
    if (model.size() < (size_t) nodeid) {
      model.resize(nodeid, R_NilValue);
    }
    model[nodeid - 1] = value;
  };
  
  // Utility lambda: scan for the next available slot in model (1-indexed).
  auto findNextSlot = [&model]() -> int {
    int j = 1;
    while(j <= (int) model.size() && (model[j-1] != R_NilValue)) {
      j++;
    }
    return j;
  };
  
  // Use a vector as a stack of nodes to process.
  vector<TreeNode> node_stack;
  node_stack.push_back(TreeNode(1, Index));
  
  // Main iterative loop.
  while (!node_stack.empty()) {
    // Pop the last node (LIFO order).
    TreeNode current = node_stack.back();
    node_stack.pop_back();
    
    int nodeid = current.nodeid;
    NumericVector indices = current.indices;
    
    if (indices.size() > (unsigned) min_leaf) {
      // Randomly select a subset of features.
      int num_features = X.ncol();
      // Create a sequence 1, 2, ..., num_features.
      IntegerVector seq_features = seq(1, num_features); 
      
      Function sampleFunc("sample");
      // Call R's sample() to select m_feature indices.
      NumericVector ff = sampleFunc(seq_features, m_feature);
      // Sort the sampled feature indices.
      std::sort(ff.begin(), ff.end());
      
      // Call your Rcpp-based splitting function.
      List Result = splitt2_fun(X, Y, m_feature, indices, Inv_Cov_Y, Command,
                                ff, nCores);
      // Assume that the first two elements of Result are:
      //   Result[0]: left indices (Index_left)
      //   Result[1]: right indices (Index_right)
      NumericVector Index_left = Result[0];
      NumericVector Index_right = Result[1];
      
      // Determine children node IDs.
      int child1, child2;
      if (nodeid == 1) {
        child1 = 2;
        child2 = 3;
      } else {
        int j = findNextSlot();
        child1 = j;
        child2 = j + 1;
      }
      NumericVector children_ids = NumericVector::create(child1, child2);
      
      // Ensure the Result list has at least 5 elements; then record children_ids.
      if (Result.size() < 5) {
        while (Result.size() < 5) {
          Result.push_back(R_NilValue);
        }
      }
      Result[4] = children_ids;
      
      // Save the current node's split result.
      setModel(nodeid, Result);
      
      // Immediately assign the split indices for the children.
      setModel(child1, Index_left);
      setModel(child2, Index_right);
      
      // Push children onto the stack.
      // (Push right child first so that left child is processed next.)
      node_stack.push_back(TreeNode(child2, Index_right));
      node_stack.push_back(TreeNode(child1, Index_left));
      
    } else {
      // Terminal (leaf) node: extract the corresponding rows of Y.
      int ncols = Y.ncol();
      int nrows = indices.size();
      NumericMatrix leaf_value(nrows, ncols);
      for (int i = 0; i < nrows; i++) {
        // Convert the index from R's 1-indexing to C++ 0-indexing.
        int row_index = static_cast<int>(indices[i]) - 1;
        for (int j = 0; j < ncols; j++) {
          leaf_value(i, j) = Y(row_index, j);
        }
      }
      // Wrap the matrix in a List (mimicking your original R code).
      List leaf = List::create(leaf_value);
      setModel(nodeid, leaf);
    }
  }
  
  // Convert the model vector into an Rcpp List.
  int n = model.size();
  List out(n);
  for (int i = 0; i < n; i++) {
    out[i] = model[i];
  }
  
  return out;
}


List build_single_tree_cpp(const NumericMatrix &X, 
                           const NumericMatrix &Y,
                           int m_feature, 
                           int min_leaf, 
                           const NumericMatrix &Inv_Cov_Y,
                           int Command,
                           int nCores = 0) {
  // Create an index vector from 1 to nrow(X)
  int n = X.nrow();
  NumericVector Index(n);
  for (int i = 0; i < n; i++) {
    Index[i] = i + 1;  // 1-indexed
  }
  
  // Call the iterative splitting function.
  List model = split_node_iterative_cpp(X, Y, m_feature, Index, min_leaf, Inv_Cov_Y, Command, nCores);
  
  return model;
}


NumericMatrix predicting_cpp(List Single_Model, int start_node, NumericVector X_test, int Variable_number) {
  int current_node = start_node;
  
  while (true) {
    // Retrieve the node (note: Single_Model is 0-indexed, but our node numbering is 1-indexed)
    List current = Single_Model[current_node - 1];
    
    // Check for an internal (split) node: we assume that internal nodes have list length equal to 5
    if (current.size() == 5) {
      // Extract the splitting feature number (third element; 1-indexed)
      int feature_no = as<int>(current[2]);
      
      // Retrieve the test observation's value for that feature (adjust for 0-indexing)
      double feature_value = X_test[feature_no - 1];
      
      // Retrieve the threshold (fourth element)
      double threshold = as<double>(current[3]);
      
      // Retrieve the children node indices (fifth element)
      NumericVector children = current[4];
      
      // Traverse to left or right child based on comparison
      if (feature_value < threshold) {
        current_node = static_cast<int>(children[0]);
      } else {
        current_node = static_cast<int>(children[1]);
      }
    } 
    else {
      // We are at a leaf node.
      // The leaf node's first element is assumed to be a numeric matrix with the responses.
      NumericMatrix leaf_matrix = current[0];
      int nrows = leaf_matrix.nrow();
      
      // Prepare a 1 x Variable_number result.
      NumericMatrix result(1, Variable_number);
      
      // Use Kahan summation for each response column.
      if (Variable_number > 1) {
        for (int j = 0; j < Variable_number; j++) {
          double sum = 0.0, c = 0.0; // c is the compensation for lost low-order bits.
          for (int r = 0; r < nrows; r++) {
            double value = leaf_matrix(r, j);
            double y = value - c;
            double t = sum + y;
            c = (t - sum) - y;
            sum = t;
          }
          result(0, j) = sum / nrows;
        }
      } 
      else {
        double sum = 0.0, c = 0.0;
        for (int r = 0; r < nrows; r++) {
          double value = leaf_matrix(r, 0);
          double y = value - c;
          double t = sum + y;
          c = (t - sum) - y;
          sum = t;
        }
        result(0, 0) = sum / nrows;
      }
      
      return result;
    } // end leaf node branch
  } // end while loop (traversal)
}

struct PredictionWorker : public Worker {
  const List Single_Model;
  const RMatrix<double> X_test;
  const int Variable_number;
  RMatrix<double> out;
  
  // Constructor
  PredictionWorker(const List Single_Model, const NumericMatrix & X_test,
                   int Variable_number, NumericMatrix out)
    : Single_Model(Single_Model), X_test(X_test),
      Variable_number(Variable_number), out(out) { }
  
  // Process each test observation from 'begin' to 'end'.
  void operator()(std::size_t begin, std::size_t end) {
    int ncols = X_test.ncol();
    for (std::size_t i = begin; i < end; i++) {
      // Read the i-th test observation into a vector.
      NumericVector obs(ncols);
      for (int j = 0; j < ncols; j++) {
        obs[j] = X_test(i, j);
      }
      // Compute prediction using the serial iterative function.
      int current_node = 1;
      NumericVector prediction(Variable_number);
      while (true) {
        List current = Single_Model[current_node - 1];
        if ( current.size() == 5 ) {
          int feature_no = as<int>( current[2] );
          double threshold = as<double>( current[3] );
          double fv = obs[ feature_no - 1 ];
          NumericVector children = current[4];
          if ( fv < threshold )
            current_node = static_cast<int>( children[0] );
          else
            current_node = static_cast<int>( children[1] );
        } else {
          NumericMatrix leaf_matrix = current[0];
          int nrows = leaf_matrix.nrow();
          if (Variable_number > 1) {
            for (int j = 0; j < Variable_number; j++) {
              double sum = 0.0;
              for (int r = 0; r < nrows; r++) {
                sum += leaf_matrix(r, j);
              }
              prediction[j] = sum / nrows;
            }
          } else {
            double sum = 0.0;
            for (int r = 0; r < nrows; r++) {
              sum += leaf_matrix(r, 0);
            }
            prediction[0] = sum / nrows;
          }
          break;
        }
      }
      // Write prediction to the output matrix.
      for (int j = 0; j < Variable_number; j++) {
        out(i, j) = prediction[j];
      }
    }
  }
};

//------------------------------------------------------------------------------
// Parallel Prediction Function
//
// This function computes predictions for all test observations.
// If nCores <= 0, it runs serially; otherwise, it uses RcppParallel's parallelFor
// with TBB's global_control to set the maximum number of threads.
//------------------------------------------------------------------------------

struct Node {
  bool isLeaf;
  int feature_no;    // valid if !isLeaf (1-indexed).
  double threshold;  // valid if !isLeaf.
  int left_child;    // valid if !isLeaf (1-indexed).
  int right_child;   // valid if !isLeaf (1-indexed).
  
  // For leaf nodes.
  std::vector<double> leaf_data;  
  int nrows;  // number of rows in the leaf matrix.
  int ncols;  // number of columns in the leaf matrix.
};

//---------------------------------------------------------------------
// Convert the R model (Rcpp List) into a vector of native Node objects.
// Crucial note: for leaf nodes, we copy the NumericMatrix data into a 
// plain std::vector to avoid accessing R objects in parallel.
//---------------------------------------------------------------------
std::vector<Node> convertModel(const List& Single_Model) {
  int nNodes = Single_Model.size();
  std::vector<Node> tree;
  tree.reserve(nNodes);
  
  for (int i = 0; i < nNodes; i++) {
    List node = Single_Model[i];
    Node cur;
    
    // Internal node: we expect the list to have size 5.
    if (node.size() == 5) {
      cur.isLeaf = false;
      cur.feature_no = as<int>(node[2]);      // splitting feature (1-indexed)
      cur.threshold = as<double>(node[3]);      // the threshold
      NumericVector children = node[4];         // children indices (1-indexed)
      cur.left_child = static_cast<int>(children[0]);
      cur.right_child = static_cast<int>(children[1]);
      // Set leaf dimensions to 0.
      cur.nrows = 0;
      cur.ncols = 0;
    } else {
      // Leaf node: we assume the leaf matrix is stored in the first element.
      cur.isLeaf = true;
      NumericMatrix leaf_matrix = node[0];
      cur.nrows = leaf_matrix.nrow();
      cur.ncols = leaf_matrix.ncol();
      cur.leaf_data.resize(cur.nrows * cur.ncols);
      // Copy the data into the native vector.
      for (int r = 0; r < cur.nrows; r++){
        for (int j = 0; j < cur.ncols; j++){
          cur.leaf_data[r * cur.ncols + j] = leaf_matrix(r, j);
        }
      }
      // Set dummy values for internal node members.
      cur.feature_no = 0;
      cur.threshold = 0.0;
      cur.left_child = 0;
      cur.right_child = 0;
    }
    tree.push_back(cur);
  }
  
  return tree;
}

//---------------------------------------------------------------------
// A parallel worker that uses the native tree structure.
// This worker will traverse the tree for each observation using only 
// native C++ containers (and not R objects), ensuring thread-safety.
//---------------------------------------------------------------------
struct NativePredictionWorker : public Worker {
  const std::vector<Node>& tree;   // Native tree.
  const RMatrix<double> X_test;    // Test observations.
  int Variable_number;             // Number of response variables.
  RMatrix<double> out;             // Output predictions.
  
  // Constructor.
  NativePredictionWorker(const std::vector<Node>& tree,
                         const NumericMatrix & X_test,
                         int Variable_number,
                         NumericMatrix out) 
    : tree(tree), X_test(X_test), Variable_number(Variable_number), out(out) { }
  
  void operator()(std::size_t begin, std::size_t end) {
    int ncols = X_test.ncol();
    
    for (std::size_t i = begin; i < end; i++) {
      // Copy the i-th test observation into a native vector.
      std::vector<double> obs(ncols);
      for (int j = 0; j < ncols; j++) {
        obs[j] = X_test(i, j);
      }
      
      // Start at the root node. Our native tree is 0-indexed.
      int current_index = 0;
      
      while (true) {
        const Node &node = tree[current_index];
        if (!node.isLeaf) {
          // Internal node: decide left/right based on feature value.
          double feature_value = obs[node.feature_no - 1];  // 1-indexed feature.
          if (feature_value < node.threshold)
            current_index = node.left_child - 1;   // Convert to 0-index.
          else
            current_index = node.right_child - 1;
        } else {
          // Leaf node: compute the mean of each response column using Kahan summation.
          for (int j = 0; j < Variable_number && j < node.ncols; j++) {
            double sum = 0.0, c = 0.0;
            for (int r = 0; r < node.nrows; r++){
              double value = node.leaf_data[r * node.ncols + j];
              double y = value - c;
              double t = sum + y;
              c = (t - sum) - y;
              sum = t;
            }
            out(i, j) = sum / node.nrows;
          }
          break; // Exit the while loop.
        }
      } // end while
    } // end for each observation
  } // end operator()
};


//---------------------------------------------------------------------
// A parallel prediction function that converts the R model
// into a native tree and uses the safe prediction worker.

NumericMatrix predicting_parallel_native_cpp(List Single_Model,
                                             NumericMatrix X_test, 
                                             int Variable_number, 
                                             int nCores = 0) {
  // Convert the model to a pure C++ tree.
  std::vector<Node> nativeTree = convertModel(Single_Model);
  
  int nObs = X_test.nrow();
  NumericMatrix out(nObs, Variable_number);
  
#if RCPP_PARALLEL_USE_TBB
  if (nCores > 0) {
    // Limit the maximum number of threads.
    tbb::global_control gc(tbb::global_control::max_allowed_parallelism, nCores);
    NativePredictionWorker worker(nativeTree, X_test, Variable_number, out);
    parallelFor(0, nObs, worker);
  } else {
    // Serial case.
    int ncols = X_test.ncol();
    for (int i = 0; i < nObs; i++) {
      NumericVector obs(ncols);
      for (int j = 0; j < ncols; j++) {
        obs[j] = X_test(i, j);
      }
      NumericMatrix pred = predicting_cpp(Single_Model, 1, obs,
                                          Variable_number);
      for (int j = 0; j < Variable_number; j++) {
        out(i, j) = pred(0, j);
      }
    }
  }
#else
  // Serial case.
  int ncols = X_test.ncol();
  for (int i = 0; i < nObs; i++) {
    NumericVector obs(ncols);
    for (int j = 0; j < ncols; j++) {
      obs[j] = X_test(i, j);
    }
    NumericMatrix pred = predicting_cpp(Single_Model, 1, obs,
                                        Variable_number);
    for (int j = 0; j < Variable_number; j++) {
      out(i, j) = pred(0, j);
    }
  }
#endif
  return out;
}



NumericMatrix single_tree_prediction_cpp(List Single_Model,
                                         NumericMatrix X_test,
                                         int Variable_number) {
  // Number of test observations:
  int nObs = X_test.nrow();
  
  // Create an output matrix to store predictions.
  NumericMatrix Y_pred(nObs, Variable_number);
  
  // Loop over each test observation.
  for (int i = 0; i < nObs; i++) {
    // Extract the i-th test observation as a vector.
    // (Note: X_test.row(i) returns a Row vector that can be used as a
    // NumericVector.)
    NumericVector xt = X_test.row(i);
    
    // Use the predicting_cpp() function starting at node 1.
    // predicting_cpp returns a 1 x Variable_number NumericMatrix.
    NumericMatrix Result_temp = predicting_cpp(Single_Model, 1, xt, Variable_number);
    
    // Copy the prediction into the i-th row of Y_pred.
    for (int j = 0; j < Variable_number; j++) {
      Y_pred(i, j) = Result_temp(0, j);
    }
  }
  
  return Y_pred;
}

//-------------------------------------------------------------
// Compute the sample covariance matrix from scratch.
// X is an n x p NumericMatrix. We compute
//    cov(i,j) = sum_{k=1}^{n} (X[k,i] - mean[i]) * (X[k,j] - mean[j]) / (n-1)
// using the underlying column–major storage and square–bracket indexing.
//-------------------------------------------------------------
NumericMatrix myCovariance(NumericMatrix X) {
  int n = X.nrow();
  int p = X.ncol();
  if(n < 2) stop("Need at least 2 observations.");
  
  NumericVector means(p);
  
  // Compute column means.
  for (int j = 0; j < p; j++){
    double s = 0.0;
    for (int i = 0; i < n; i++){
      s += X[i + j * n];  // Use square bracket indexing.
    }
    means[j] = s / n;
  }
  
  // Compute covariance.
  NumericMatrix covMat(p, p); // p x p matrix.
  for (int i = 0; i < p; i++){
    for (int j = i; j < p; j++){
      double s = 0.0;
      for (int k = 0; k < n; k++){
        double d1 = X[k + i * n] - means[i];
        double d2 = X[k + j * n] - means[j];
        s += d1 * d2;
      }
      double cov = s / (n - 1);
      covMat[i + j * p] = cov;   // element (i, j)
      if(i != j)
        covMat[j + i * p] = cov; // symmetric assignment at (j, i)
    }
  }
  
  return covMat;
}

//-------------------------------------------------------------
// Compute the Cholesky decomposition of A (n x n) from scratch.
// A must be symmetric and positive–definite. Returns a lower 
// triangular matrix L such that A = L * t(L).  
// We use square–bracket indexing throughout.
//-------------------------------------------------------------
NumericMatrix cholDecomp(NumericMatrix A) {
  int n = A.nrow();
  if(n != A.ncol()) stop("Matrix is not square.");
  
  NumericMatrix L(n, n);
  // Initialize L to zero (by default NumericMatrix values are 0, but we do it explicitly).
  for (int i = 0; i < n * n; i++){
    L[i] = 0.0;
  }
  
  for (int i = 0; i < n; i++){
    double sum = A[i + i * n];  // diagonal element A(i,i)
    for (int k = 0; k < i; k++){
      double Lik = L[i + k * n];  // L(i,k)
      sum -= Lik * Lik;
    }
    if(sum <= 0) stop("Matrix is not positive definite.");
    L[i + i * n] = std::sqrt(sum);
    for (int j = i + 1; j < n; j++){
      double sum2 = A[j + i * n];  // A(j,i)
      for (int k = 0; k < i; k++){
        sum2 -= L[j + k * n] * L[i + k * n];
      }
      L[j + i * n] = sum2 / L[i + i * n];
    }
  }
  
  return L;
}

//-------------------------------------------------------------
// Invert a lower–triangular matrix L (n x n) using forward substitution.
// Returns Linv such that L * Linv = identity.
//-------------------------------------------------------------
NumericMatrix invLowerTri(const NumericMatrix &L) {
  int n = L.nrow();
  NumericMatrix Linv(n, n);
  
  // Initialize Linv to all zeros.
  for (int i = 0; i < n * n; i++){
    Linv[i] = 0.0;
  }
  
  // Invert L.
  for (int i = 0; i < n; i++){
    int i_n = i * n;
    Linv[i + i_n] = 1.0 / L[i + i_n];
    for (int j = i + 1; j < n; j++){
      double sum = 0.0;
      for (int k = i; k < j; k++){
        sum += L[j + k * n] * Linv[k + i_n];
      }
      Linv[j + i_n] = -sum / L[j + j * n];
    }
  }
  return Linv;
}

//-------------------------------------------------------------
// Transpose a numeric matrix A using square–bracket indexing.
// A has dimensions n x m. Returns T, an m x n matrix where T(j,i) = A(i,j).
//-------------------------------------------------------------
NumericMatrix transposeMat(const NumericMatrix &A) {
  int n = A.nrow();
  int m = A.ncol();
  NumericMatrix T(m, n);
  for (int i = 0; i < n; i++){
    int i_m = i * m;
    for (int j = 0; j < m; j++){
      T[j + i_m] = A[i + j * n];
    }
  }
  return T;
}

//-------------------------------------------------------------
// Multiply matrices A (n x m) and B (m x p) using square–bracket indexing.
// Returns C (n x p), where C(i,j) = sum_{k} A(i,k) * B(k,j).
//-------------------------------------------------------------
NumericMatrix matMultiply(const NumericMatrix &A, const NumericMatrix &B) {
  int n = A.nrow();
  int m = A.ncol();
  int p = B.ncol();
  if(m != B.nrow()) stop("Incompatible dimensions for multiplication.");
  
  NumericMatrix C(n, p);
  // Initialize C to zeros.
  for (int i = 0; i < n * p; i++) {
    C[i] = 0.0;
  }
  
  for (int i = 0; i < n; i++){
    for (int j = 0; j < p; j++){
      double sum = 0.0;
      int j_m = j * m;
      for (int k = 0; k < m; k++){
        sum += A[i + k * n] * B[k + j_m];
      }
      C[i + j * n] = sum;
    }
  }
  return C;
}

//-------------------------------------------------------------
// Compute the inverse of a symmetric positive–definite matrix A using
// our own Cholesky–based implementation.
// Steps:
//   1. Compute the Cholesky decomposition: A = L * t(L).
//   2. Invert L (lower triangular).
//   3. Compute Ainv = transpose(Linv) * Linv.
//-------------------------------------------------------------
NumericMatrix myInvSympd(NumericMatrix A) {
  int n = A.nrow();
  if(n != A.ncol()) stop("Input matrix A must be square.");
  
  // Step 1: Cholesky decomposition.
  NumericMatrix L = cholDecomp(A);
  
  // Step 2: Invert L.
  NumericMatrix Linv = invLowerTri(L);
  
  // Step 3: Compute Ainv = t(Linv) * Linv.
  NumericMatrix LinvT = transposeMat(Linv);
  NumericMatrix Ainv = matMultiply(LinvT, Linv);
  
  return Ainv;
}

// ================================================================
// 1. Parallel Covariance function using square–bracket indexing
// ================================================================

// Worker for computing the covariance for a range of columns.
// Here X is assumed to be of dimensions n x p in column–major format.
// We assume the means vector has been computed already.
struct CovWorker : public Worker {
  int n;                // number of observations (rows)
  int p;                // number of variables (columns)
  const double* X;      // pointer to X data
  const double* means;  // pointer to precomputed means
  double* covMat;       // pointer to covariance matrix
  // ...
  CovWorker(const NumericMatrix &Xmat, int n_, int p_,
            const NumericVector &meansVec, NumericMatrix &covMatMat)
    : n(n_), p(p_), X(&(Xmat[0])), means(&(meansVec[0])),
      covMat(&(covMatMat[0])) { }
  
  // Operator over a range of "i" (columns). For each i, we compute for j = i, ... , p–1.
  void operator()(std::size_t begin, std::size_t end) {
    for(std::size_t i = begin; i < end; i++){
      int i_n = i * n;
      for (int j = i; j < p; j++){
        double s = 0.0;
        int j_n = j * n;
        for (int k = 0; k < n; k++){
          double d1 = X[k + i_n] - means[i];
          double d2 = X[k + j_n] - means[j];
          s += d1 * d2;
        }
        double cov = s / (n - 1);
        covMat[i + j * p] = cov;    // store (i,j)
        if (i != (size_t)j)
          covMat[j + i * p] = cov;  // mirror to (j,i)
      }
    }
  }
};

NumericMatrix myCovarianceParallel(NumericMatrix X, int nCores = 0) {
  int n = X.nrow();
  int p = X.ncol();
  if(n < 2) stop("Need at least 2 observations.");
  
  // Compute the column means (serially)
  NumericVector means(p);
  for (int j = 0; j < p; j++){
    double s = 0.0;
    int j_n = j * n;
    for (int i = 0; i < n; i++){
      s += X[i + j_n];  // square–bracket indexing
    }
    means[j] = s / n;
  }
  
  NumericMatrix covMat(p, p); // p x p output
  
  CovWorker worker(X, n, p, means, covMat);
  if(nCores > 0)
    parallelFor(0, (size_t)p, worker, nCores);
  else
    worker(0, p);
  
  return covMat;
}

// ================================================================
// 2. Parallel Inverse of an SPD matrix using a Cholesky approach
// ================================================================
//
// We implement the following helper functions using square–bracket indexing:
//
//   (a) cholDecomp()   -- Cholesky decomposition (sequential)
//   (b) invLowerTri()  -- Inversion of a lower–triangular matrix (sequential)
//   (c) transposeMat() -- Transposition (sequential)
//   (d) MatMulWorker   -- Parallel matrix multiplication worker for computing C = A * B
//   (e) myInvSympdParallel() -- Computes the inverse from A = L * t(L)
//

// (a) Cholesky decomposition: For symmetric positive–definite A, 
// return L (lower triangular) with A = L * t(L).

// (b) Inversion of a lower triangular matrix L using forward substitution.
// Returns Linv such that L * Linv = Identity.

// (c) Transposition of a matrix A (n x m). Returns T (m x n)

// (d) Parallel Matrix Multiplication Worker.
// Computes C = A * B, where A is (n x m) and B is (m x p), storing result in C (n x p).
// We use square–bracket indexing, assuming column–major storage.
struct MatMulWorker : public Worker {
  int n, m, p;     // dimensions: A: n x m, B: m x p, C: n x p.
  const double* A; // pointer to matrix A
  const double* B; // pointer to matrix B
  double* C;       // pointer to result matrix C
  
  MatMulWorker(const NumericMatrix &A_, const NumericMatrix &B_, NumericMatrix &C_)
    : n(A_.nrow()), m(A_.ncol()), p(B_.ncol()),
      A(&(A_[0])), B(&(B_[0])), C(&(C_[0])) { }
  
  void operator()(std::size_t begin, std::size_t end) {
    for (std::size_t i = begin; i < end; i++){
      for (int j = 0; j < p; j++){
        double sum = 0.0;
        for (int k = 0; k < m; k++){
          sum += A[i + k * n] * B[k + j * m];
        }
        C[i + j * n] = sum;
      }
    }
  }
};

// (e) myInvSympdParallel: Computes the inverse of SPD matrix A.
// Here we compute L = cholDecomp(A), then Linv = invLowerTri(L), then 
// Ainv = transpose(Linv) * Linv, where the final multiplication is
// parallelized using MatMulWorker.
NumericMatrix myInvSympdParallel(NumericMatrix A, int nCores = 0) {
  int n = A.nrow();
  if(n != A.ncol()) stop("Input matrix must be square.");
  
  NumericMatrix L = cholDecomp(A);       // Sequential Cholesky.
  NumericMatrix Linv = invLowerTri(L);     // Sequential inversion.
  NumericMatrix LinvT = transposeMat(Linv); // Sequential transpose.
  
  NumericMatrix Ainv(n, n);
  MatMulWorker worker(LinvT, Linv, Ainv);
  if(nCores > 0)
    parallelFor(0, (size_t)n, worker, nCores);
  else
    worker(0, n);
  
  return Ainv;
}

NumericMatrix cov_fun(NumericMatrix X, int nCores = 0) {
  
#if RCPP_PARALLEL_USE_TBB
  if (nCores > 0){
    return myCovarianceParallel(X, nCores);
  } else {
    return myCovariance(X);
  }
#else
  return myCovariance(X);
#endif
}

NumericMatrix InvSympd_fun(NumericMatrix X, int nCores = 0) {
  
#if RCPP_PARALLEL_USE_TBB
  if (nCores > 0){
    return myInvSympdParallel(X, nCores);
  } else {
    return myInvSympd(X);
  }
#else
  return myInvSympd(X);
#endif
}

// [[Rcpp::export(".build_forest_predict_cpp")]]
NumericMatrix build_forest_predict_cpp(NumericMatrix trainX, NumericMatrix trainY, 
                                       int n_tree, int m_feature, int min_leaf, 
                                       NumericMatrix testX, int nCores = 0) {
  // Parameter checks.
  int n_train = trainX.nrow();
  int p_train = trainX.ncol();
  if(n_tree < 1)
    stop("Number of trees in the forest must be at least 1");
  if(m_feature < 1)
    stop("Number of randomly selected features for a split must be at least 1");
  if(min_leaf < 1 || min_leaf > n_train)
    stop("Minimum leaf number must be at least 1 and no larger than the number of training samples");
  if(p_train != testX.ncol() || n_train != trainY.nrow())
    stop("Data size is inconsistent");
  
  // Determine number of responses & test sample size.
  int Variable_number = trainY.ncol();
  int n_test = testX.nrow();
  
  // Preallocate output matrix Y_HAT (predictions) and initialize all entries to zero.
  NumericMatrix Y_HAT(n_test, Variable_number);
  std::fill(Y_HAT.begin(), Y_HAT.end(), 0.0);
  
  // Loop over each tree.
  for (int tree_index = 0; tree_index < n_tree; tree_index++){
    // ---------------------------------------------------------------------
    // Bootstrap Sample Generation using Rcpp's sample() sugar
    // This generates a sample over 1:n_train (one-based) of size n_train with replacement.
    IntegerVector idx = sample(n_train, n_train, true);
    // Convert to std::vector<int> and adjust to zero-based indexing.
    std::vector<int> indices = as< std::vector<int> >(idx);
    for (size_t i = 0; i < indices.size(); i++){
      indices[i] = indices[i] - 1;
    }
    // ---------------------------------------------------------------------
    
    // Subset the training sets.
    // X_sub will be an n_train x p_train matrix,
    // Y_sub will be an n_train x Variable_number matrix.
    NumericMatrix X_sub(n_train, p_train);
    NumericMatrix Y_sub(n_train, Variable_number);
    
    // Subset trainX.
    for (int j = 0; j < p_train; j++){
      for (int i = 0; i < n_train; i++){
        X_sub(i, j) = trainX(indices[i], j);
      }
    }
    // Subset trainY.
    for (int j = 0; j < Variable_number; j++){
      for (int i = 0; i < n_train; i++){
        Y_sub(i, j) = trainY(indices[i], j);
      }
    }
    
    // Determine model command: if multivariate responses, use Command 2.
    int Command = (Variable_number > 1) ? 2 : 1;
    
    // Compute the inverse covariance matrix if needed.
    NumericMatrix InvCovY;
    if (Command == 2) {
      // Use your custom myCovariance function to compute the covariance matrix.
      // NumericMatrix covMat = cov_fun(Y_sub);
      // Compute its inverse using your myInvSympd function.
      InvCovY = InvSympd_fun(cov_fun(Y_sub));//InvSympd_fun(covMat);
    } else {
      InvCovY = NumericMatrix(2, 2); // Dummy matrix in the univariate case.
    }
    
    // Build a single tree using the bootstrap sample.
    List Single_Model = build_single_tree_cpp(X_sub, Y_sub, m_feature, min_leaf, 
                                              InvCovY, Command, nCores);
    
    // Get predictions on the test set from this tree.
    NumericMatrix Y_pred = predicting_parallel_native_cpp(Single_Model, testX, Variable_number, nCores);
    
    // Accumulate predictions from this tree.
    int totElems = Y_HAT.size();
    for (int i = 0; i < totElems; i++){
      Y_HAT[i] += Y_pred[i];
    }
  }
  
  // Average the predictions over all trees.
  int totElems = Y_HAT.size();
  for (int i = 0; i < totElems; i++){
    Y_HAT[i] /= n_tree;
  }
  
  return Y_HAT;
}
