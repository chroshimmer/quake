//
// Created by Jason on 9/11/24.
// Prompt for GitHub Copilot:
// - Conform to the google style guide
// - Use descriptive variable names

#ifndef LIST_SCANNING_H
#define LIST_SCANNING_H

#include <common.h>
#include "faiss/utils/Heap.h"
#include "faiss/utils/distances.h"



#ifdef QUAKE_ENABLE_GPU
#include <cuvs/distance/distance.hpp>
#include <cuvs/selection/select_k.hpp>
#include <raft/core/resources.hpp>

#include <c10/cuda/CUDAStream.h>
#include <raft/core/device_mdspan.hpp> // RAFT device view (make_device_matrix_view, etc.)
#endif

inline Tensor calculate_recall(Tensor ids, Tensor gt_ids) {
    Tensor num_correct = torch::zeros(ids.size(0), torch::kInt64);
    int num_queries = ids.size(0);
    int k = ids.size(1);

    int64_t *ids_ptr = ids.data_ptr<int64_t>();
    int64_t *gt_ids_ptr = gt_ids.data_ptr<int64_t>();

    for (int i = 0; i < num_queries; i++) {
        std::unordered_set<int64_t> gt_label_set;
        for (int j = 0; j < k; j++) {
            gt_label_set.insert(gt_ids_ptr[i * k + j]);
        }
        for (int j = 0; j < k; j++) {
            if (gt_label_set.find(ids_ptr[i * k + j]) != gt_label_set.end()) {
                num_correct[i] += 1;
            }
        }
    }

    Tensor recall = num_correct.to(torch::kFloat32) / k;

    return recall;
}

#define TOP_K_BUFFER_CAPACITY (8 * 1024)

template<typename DistanceType = float, typename IdType = int>
class TypedTopKBuffer {
public:
    int k_; // Number of top elements to keep
    int curr_offset_ = 0; // Current offset in the buffer
    std::vector<std::pair<DistanceType, IdType> > topk_; // Buffer to store top-k elements
    bool is_descending_; // Flag to indicate sorting order
    std::recursive_mutex buffer_mutex_;
    std::atomic<bool> processing_query_;
    std::atomic<int> jobs_left_;
    std::atomic<int> partitions_scanned_;

    TypedTopKBuffer(int k, bool is_descending, int buffer_capacity = TOP_K_BUFFER_CAPACITY)
        : k_(k), is_descending_(is_descending), topk_(buffer_capacity), processing_query_(true), partitions_scanned_(0) {
        assert(k <= buffer_capacity); // Ensure k is smaller than or equal to buffer size

        for (int i = 0; i < topk_.size(); i++) {
            if (is_descending_) {
                topk_[i] = {-std::numeric_limits<DistanceType>::infinity(), -1};
            } else {
                topk_[i] = {std::numeric_limits<DistanceType>::max(), -1};
            }
        }
    }

    ~TypedTopKBuffer() = default;

    void set_k(int new_k) {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        assert(new_k <= topk_.size());
        k_ = new_k;
        reset();
    }

    void set_processing_query(bool new_value) {
        processing_query_.store(new_value, std::memory_order_relaxed);
    }

    inline bool currently_processing_query() {
        return processing_query_.load(std::memory_order_relaxed);
    }

    void set_jobs_left(int total_jobs) {
        jobs_left_.store(total_jobs, std::memory_order_relaxed);
    }

    void record_skipped_jobs(int skipped_jobs) {
        jobs_left_.fetch_sub(skipped_jobs, std::memory_order_relaxed);
    }

    void record_empty_job() {
        jobs_left_.fetch_sub(1, std::memory_order_relaxed);
    }

    inline bool finished_all_jobs() {
        int curr_jobs_left = jobs_left_.load(std::memory_order_relaxed);
        return jobs_left_.load(std::memory_order_relaxed) <= 0;
    }

    inline int get_num_partitions_scanned() {
        return partitions_scanned_.load(std::memory_order_relaxed);
    }

    void reset() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        curr_offset_ = 0;
        for (int i = 0; i < k_; i++) {
            if (is_descending_) {
                topk_[i] = { -std::numeric_limits<DistanceType>::infinity(), -1 };
            } else {
                topk_[i] = { std::numeric_limits<DistanceType>::max(), -1 };
            }
        }
        partitions_scanned_.store(0, std::memory_order_relaxed);
    }

    void add(DistanceType distance, IdType index) {
        if (curr_offset_ >= topk_.size()) {
            flush(); // Flush the buffer if it is full
        }
        topk_[curr_offset_++] = {distance, index};
    }

    void batch_add(DistanceType *distances, const IdType *indices, int num_values) {
        if (num_values == 0) {
            jobs_left_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
        if (!currently_processing_query()) {
            jobs_left_.fetch_sub(1, std::memory_order_relaxed);
            return;
        }
        std::lock_guard<std::recursive_mutex> lock(buffer_mutex_);
        int pos = 0;
        while (pos < num_values) {
            int available = static_cast<int>(topk_.size()) - curr_offset_;
            if (available <= 0) {
                flush();
                available = static_cast<int>(topk_.size()) - curr_offset_;
            }
            int to_copy = std::min(num_values - pos, available);
            for (int i = 0; i < to_copy; i++) {
                topk_[curr_offset_++] = { distances[pos + i], indices[pos + i] };
            }
            pos += to_copy;
        }
        jobs_left_.fetch_sub(1, std::memory_order_relaxed);
        partitions_scanned_.fetch_add(1, std::memory_order_relaxed);
    }

    DistanceType flush() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        if (curr_offset_ > k_) {
            if (is_descending_) {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto &a, const auto &b) { return a.first > b.first; });
            } else {
                std::partial_sort(topk_.begin(), topk_.begin() + k_, topk_.begin() + curr_offset_,
                                  [](const auto &a, const auto &b) { return a.first < b.first; });
            }
            curr_offset_ = k_; // After flush, retain only the top-k elements
        } else {
            // sort the curr_offset_ elements
            if (is_descending_) {
                std::sort(topk_.begin(), topk_.begin() + curr_offset_,
                          [](const auto &a, const auto &b) { return a.first > b.first; });
            } else {
                std::sort(topk_.begin(), topk_.begin() + curr_offset_,
                          [](const auto &a, const auto &b) { return a.first < b.first; });
            }
        }
        return topk_[std::min(curr_offset_, k_ - 1)].first;
    }

    std::vector<DistanceType> get_topk() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed

        std::vector<DistanceType> topk_distances(std::min(curr_offset_, k_));
        for (int i = 0; i < std::min(curr_offset_, k_); i++) {
            topk_distances[i] = topk_[i].first;
        }

        return topk_distances;
    }

    DistanceType get_kth_distance() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed
        return topk_[std::min(curr_offset_, k_ - 1)].first;
    }

    // Get the current top-k indices (after final flush)
    std::vector<IdType> get_topk_indices() {
        std::lock_guard<std::recursive_mutex> buffer_lock(buffer_mutex_);
        flush(); // Ensure the buffer is properly flushed

        std::vector<IdType> topk_indices(std::min(curr_offset_, k_));
        for (int i = 0; i < std::min(curr_offset_, k_); i++) {
            topk_indices[i] = topk_[i].second;
        }
        return topk_indices;
    }
};

// Type alias for convenience
using TopkBuffer = TypedTopKBuffer<float, int64_t>;

inline std::tuple<Tensor, Tensor> buffers_to_tensor(vector<shared_ptr<TopkBuffer>> buffers) {
    int n = buffers.size();
    int k = buffers[0]->k_;
    Tensor topk_distances = torch::empty({n, k}, torch::kFloat32);
    Tensor topk_indices = torch::empty({n, k}, torch::kInt64);

    auto topk_distances_accessor = topk_distances.accessor<float, 2>();
    auto topk_indices_accessor = topk_indices.accessor<int64_t, 2>();

    for (int i = 0; i < n; i++) {
        vector<float> distances = buffers[i]->get_topk();
        vector<int64_t> indices = buffers[i]->get_topk_indices();

        int curr_k = std::min(k, (int) distances.size());

        for (int j = 0; j < curr_k; j++) {
            topk_distances_accessor[i][j] = distances[j];
            topk_indices_accessor[i][j] = indices[j];
        }
    }

    return std::make_tuple(topk_indices, topk_distances);
}

inline vector<shared_ptr<TopkBuffer>> create_buffers(int n, int k, bool is_descending) {
    vector<shared_ptr<TopkBuffer>> buffers(n);
    for (int i = 0; i < n; i++) {
        buffers[i] = make_shared<TopkBuffer>(k, is_descending, 10 * k);
    }
    return buffers;
}

inline void scan_list_no_ids_inner_product(const float *query_vec,
                                                   const float *list_vecs,
                                                   int list_size,
                                                   int d,
                                                   TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(faiss::fvec_inner_product(query_vec, vec, d), l);
        vec += d;  // move pointer to next vector
    }
}

inline void scan_list_no_ids_l2(const float *query_vec,
                                      const float *list_vecs,
                                      int list_size,
                                      int d,
                                      TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(sqrt(faiss::fvec_L2sqr(query_vec, vec, d)), l);
        vec += d;
    }
}

inline void scan_list_with_ids_inner_product(const float *query_vec,
                                                     const float *list_vecs,
                                                     const int64_t *list_ids,
                                                     int list_size,
                                                     int d,
                                                     TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(faiss::fvec_inner_product(query_vec, vec, d), list_ids[l]);
        vec += d;
    }
}

inline void scan_list_with_ids_l2(const float *query_vec,
                                        const float *list_vecs,
                                        const int64_t *list_ids,
                                        int list_size,
                                        int d,
                                        TopkBuffer &buffer) {
    const float *vec = list_vecs;
    for (int l = 0; l < list_size; l++) {
        buffer.add(sqrt(faiss::fvec_L2sqr(query_vec, vec, d)), list_ids[l]);
        vec += d;
    }
}

#ifdef QUAKE_ENABLE_GPU
inline void gpu_select_k(const float* query_vec,  // (d,)
                          const float* list_vecs,  // (list_size, d)
                          const int64_t* list_ids, // (list_size,) or nullptr
                          int list_size,
                          int d,
                          int k,
                          faiss::MetricType metric,
                          float* out_distances,     // (list_size,)
                          int64_t* out_indices,    // (k,)
                          float* out_k_distances  // (k,)
                          ) {
    // debug test
    // std::cout << "!!! In gpu_scan_list !!!" << std::endl;

    raft::resources handle;
    cudaError_t cuda_status;
    
    // Wrap device pointers with RAFT views
    auto query_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        query_vec, 1, d);
    
    auto list_view = raft::make_device_matrix_view<const float, int64_t, raft::layout_c_contiguous>(
        list_vecs, list_size, d);
    
    auto dist_view = raft::make_device_matrix_view<float, int64_t, raft::layout_c_contiguous>(
        out_distances, 1, list_size);

    // Determine distance metric
    cuvs::distance::DistanceType dist_metric = (metric == faiss::METRIC_INNER_PRODUCT)
                                   ? cuvs::distance::DistanceType::InnerProduct
                                   : cuvs::distance::DistanceType::L2SqrtExpanded;

    try {
        cudaEvent_t start;
        cudaEvent_t stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // Compute pairwise distances (1 query vs list)
        cuvs::distance::pairwise_distance(handle, query_view, list_view, dist_view, dist_metric);
        
        // Synchronize to catch any errors from pairwise_distance
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA error after pairwise_distance: " + 
                                     std::string(cudaGetErrorString(cuda_status)));
        }

        // Now select top-k from the result row
        auto in_val = raft::make_device_matrix_view<const float, int64_t, raft::row_major>(
            out_distances, 1, list_size);
        auto out_val = raft::make_device_matrix_view<float, int64_t, raft::row_major>(
            out_k_distances, 1, k);
        auto out_idx = raft::make_device_matrix_view<int64_t, int64_t, raft::row_major>(
            out_indices, 1, k);

        // Whether we want min or max depends on the metric
        bool select_min = (metric != faiss::METRIC_INNER_PRODUCT);

        cuvs::selection::select_k(handle, in_val, std::nullopt, out_val, out_idx, select_min);
        
        // Synchronize to catch any errors from select_k
        cuda_status = cudaDeviceSynchronize();
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("CUDA error after select_k: " + 
                                     std::string(cudaGetErrorString(cuda_status)));
        }

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);

        std::cout << "[Actual search time in gpu_scan_list]" << ms << " ms" << std::endl;

    } catch (const std::exception& e) {
        // Cleanup any CUDA errors to avoid affecting other operations
        cudaGetLastError();
        throw;
    }
}

inline void gpu_scan_list(
    const float *query_vec,
    const float *list_vecs,
    const int64_t *list_ids,
    int list_size,
    int d,
    TopkBuffer &buffer,
    faiss::MetricType metric = faiss::METRIC_L2) {

    if (list_size <= 0) {
        return;
    }
    cudaError_t cuda_status;
    // Call the GPU implementation
    int k = std::min(list_size, buffer.k_); 
    
    // Allocate device memory for inputs
    float* d_query = nullptr;
    
    // Allocate device memory for results
    float* d_distances = nullptr;
    int64_t* d_indices = nullptr;
    float* d_k_distances = nullptr;
    
    try {
        // Allocate and copy query vector to device
        cuda_status = cudaMalloc(&d_query, sizeof(float) * d);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for query: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        cuda_status = cudaMemcpy(d_query, query_vec, sizeof(float) * d, cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy query to device: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        // Allocate device memory for results
        cuda_status = cudaMalloc(&d_distances, sizeof(float) * list_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for distances: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        cuda_status = cudaMalloc(&d_indices, sizeof(int64_t) * k);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for indices: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        cuda_status = cudaMalloc(&d_k_distances, sizeof(float) * k);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to allocate device memory for k_distances: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        // Initialize output memory
        cuda_status = cudaMemset(d_distances, 0, sizeof(float) * list_size);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to initialize distances: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        cuda_status = cudaMemset(d_indices, 0, sizeof(int64_t) * k);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to initialize indices: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        cuda_status = cudaMemset(d_k_distances, 0, sizeof(float) * k);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to initialize k_distances: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Run GPU scan list
        gpu_select_k(d_query, list_vecs, list_ids,
                    list_size, d, k,
                    metric,
                    d_distances, d_indices, d_k_distances);
                    
        // Copy results back to host
        std::vector<float> h_distances(k);
        std::vector<int64_t> h_indices(k);
        
        cuda_status = cudaMemcpy(h_distances.data(), d_k_distances, sizeof(float) * k, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy distances from device: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }
        
        cuda_status = cudaMemcpy(h_indices.data(), d_indices, sizeof(int64_t) * k, cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error("Failed to copy indices from device: " + 
                                    std::string(cudaGetErrorString(cuda_status)));
        }

        // Map back to original list_ids
        if (list_ids != nullptr) {
            std::vector<int64_t> h_list_ids(list_size);
            cudaMemcpy(h_list_ids.data(), list_ids, list_size * sizeof(int64_t), cudaMemcpyDeviceToHost);
            for (int i = 0; i < k; ++i) {
                h_indices[i] = h_list_ids[h_indices[i]];
            }
        }

        //DEBUG: 
        // Print out results for debugging
        // std::cout << "Distances: ";
        // for (int i = 0; i < k; ++i) {
        //     std::cout << h_distances[i] << " ";
        // }
        // std::cout << std::endl;

        // std::cout << "Indices: ";
        // for (int i = 0; i < k; ++i) {
        //     std::cout << h_indices[i] << " ";
        // }
        // std::cout << std::endl;


        // Add results to TopkBuffer
        for (int i = 0; i < k; ++i) {
            buffer.add(h_distances[i], h_indices[i]);
        }

        // Free device memory
        if (d_query) cudaFree(d_query);
        if (d_distances) cudaFree(d_distances);
        if (d_indices) cudaFree(d_indices);
        if (d_k_distances) cudaFree(d_k_distances);

    } catch (const std::exception& e) {
        // Clean up any CUDA resources that might have been allocated
        cudaGetLastError(); // Clear any CUDA errors

        // Free device memory
        if (d_query) cudaFree(d_query);
        if (d_distances) cudaFree(d_distances);
        if (d_indices) cudaFree(d_indices);
        if (d_k_distances) cudaFree(d_k_distances);

        // Re-throw the exception with more context
        throw std::runtime_error("GPU scan_list failed: " + std::string(e.what()));
    }
}
#endif

// The main scan_list function that dispatches to one of the specialized functions.
inline void scan_list(const float *query_vec,
                     const float *list_vecs,
                     const int64_t *list_ids,
                     int list_size,
                     int d,
                     TopkBuffer &buffer,
                     faiss::MetricType metric = faiss::METRIC_L2) {

    // Dispatch based on metric type and whether list_ids is provided.
    if (metric == faiss::METRIC_INNER_PRODUCT) {
        if (list_ids == nullptr)
            scan_list_no_ids_inner_product(query_vec, list_vecs, list_size, d, buffer);
        else
            scan_list_with_ids_inner_product(query_vec, list_vecs, list_ids, list_size, d, buffer);
    } else { // Assume L2 (or similar)
        if (list_ids == nullptr)
            scan_list_no_ids_l2(query_vec, list_vecs, list_size, d, buffer);
        else
            scan_list_with_ids_l2(query_vec, list_vecs, list_ids, list_size, d, buffer);
    }
}


inline void batched_scan_list(const float *query_vecs,
                              const float *list_vecs,
                              const int64_t *list_ids,
                              int num_queries,
                              int list_size,
                              int dim,
                              vector<shared_ptr<TopkBuffer>> &topk_buffers,
                              MetricType metric = faiss::METRIC_L2) {
    if (list_size == 0 || list_vecs == nullptr) {
        // No list vectors to process;
        return;
    }

    // Ensure k does not exceed list_size
    int k = topk_buffers[0]->k_;
    int k_max = std::min(k, list_size);

    int64_t *labels = (int64_t *) malloc(num_queries * k_max * sizeof(int64_t));
    float *distances = (float *) malloc(num_queries * k_max * sizeof(float));

    if (metric == faiss::METRIC_INNER_PRODUCT) {
        faiss::float_minheap_array_t res = {size_t(num_queries), size_t(k_max), labels, distances};
        faiss::knn_inner_product(query_vecs, list_vecs, dim, num_queries, list_size, &res, nullptr);
    } else if (metric == faiss::METRIC_L2) {
        faiss::float_maxheap_array_t res = {size_t(num_queries), size_t(k_max), labels, distances};
        faiss::knn_L2sqr(query_vecs, list_vecs, dim, num_queries, list_size, &res, nullptr, nullptr);
    } else {
        throw std::runtime_error("Metric type not supported");
    }

    // map the labels to the actual list_ids
    if (list_ids != nullptr) {
        for (int i = 0; i < num_queries; i++) {
            for (int j = 0; j < k_max; j++) {
                labels[i * k_max + j] = list_ids[labels[i * k_max + j]];
            }
        }
    }

    // if the metric is l2, convert the distances to sqrt
    if (metric == faiss::METRIC_L2) {
        for (int i = 0; i < num_queries * k_max; i++) {
            distances[i] = sqrt(distances[i]);
        }
    }

    // add distances to the topk buffers
    for (int i = 0; i < num_queries; i++) {
        topk_buffers[i]->batch_add(distances + i * k_max, labels + i * k_max, k_max);
    }

    free(labels);
    free(distances);
}

// }
#endif //LIST_SCANNING_H
