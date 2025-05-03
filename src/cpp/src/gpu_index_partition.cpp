#ifdef QUAKE_ENABLE_GPU
#include <gpu_index_partition.h>

void GPUIndexPartition::free_memory() {
    if (codes_ == nullptr && ids_ == nullptr) {
        return;
    }

    cudaFree(codes_);
    cudaFree(ids_);
    codes_ = nullptr;
    ids_ = nullptr;
}

template <typename T>
T* GPUIndexPartition::allocate_memory(size_t num_elements, int /*numa_node*/) {
    size_t total_bytes = num_elements * sizeof(T);
    T* ptr = nullptr;
    
    cudaError_t cuda_status = cudaMalloc(&ptr, total_bytes);
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to allocate device memory in allocate_memory: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }
    return ptr;
}

void GPUIndexPartition::ensure_capacity(int64_t required) {
    if (required > buffer_size_) {
        int64_t new_capacity = std::max<int64_t>(1024, buffer_size_);
        while (new_capacity < required) {
            new_capacity *= 2;
        }
        reallocate_memory(new_capacity);
    }
}

GPUIndexPartition::~GPUIndexPartition() {
    clear();
}

void GPUIndexPartition::clear() {
    free_memory();
    numa_node_ = -1;
    core_id_ = -1;
    buffer_size_ = 0;
    num_vectors_ = 0;
    code_size_ = 0;
    codes_ = nullptr;
    ids_ = nullptr;
}

void GPUIndexPartition::reallocate_memory(int64_t new_capacity) {
    if (new_capacity < num_vectors_) {
        num_vectors_ = new_capacity;
    }

    const size_t code_bytes = static_cast<size_t>(code_size_);
    int64_t curr_count = num_vectors_;

    uint8_t* new_codes = allocate_memory<uint8_t>(new_capacity * code_bytes, numa_node_);
    idx_t* new_ids = allocate_memory<idx_t>(new_capacity, numa_node_);

    if (codes_ && ids_) {
        cudaError_t cuda_status = cudaMemcpy(new_codes, codes_, curr_count * code_bytes, cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed for codes_: ") + cudaGetErrorString(cuda_status));
        }

        cuda_status = cudaMemcpy(new_ids, ids_, curr_count * sizeof(idx_t), cudaMemcpyDeviceToDevice);
        if (cuda_status != cudaSuccess) {
            throw std::runtime_error(std::string("cudaMemcpy failed for ids_: ") + cudaGetErrorString(cuda_status));
        }
    }

    free_memory(); 

    codes_ = new_codes;
    ids_ = new_ids;
    buffer_size_ = new_capacity;
}

void GPUIndexPartition::append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes) {
    if (n_entry <= 0) return;
    ensure_capacity(num_vectors_ + n_entry);
    const size_t code_bytes = static_cast<size_t>(code_size_);

    cudaError_t cuda_status = cudaMemcpy(
        codes_ + num_vectors_ * code_bytes,
        new_codes,
        n_entry * code_bytes,
        cudaMemcpyDeviceToDevice
    );
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to copy new codes from device to codes device: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }

    cuda_status = cudaMemcpy(
        ids_ + num_vectors_,
        new_ids,
        n_entry * sizeof(idx_t),
        cudaMemcpyDeviceToDevice
    );
    if (cuda_status != cudaSuccess) {
        throw std::runtime_error("Failed to copy new ids from device to ids device: " + 
                                std::string(cudaGetErrorString(cuda_status)));
    }

    num_vectors_ += n_entry;
}
#endif