#ifdef QUAKE_ENABLE_GPU
#include <index_partition.h>
#include "cuda_runtime.h"

class GPUIndexPartition : public IndexPartition {
private:
    void free_memory();

    template <typename T>
    T* allocate_memory(size_t num_elements, int /*numa_node*/);

    void ensure_capacity(int64_t required);
public:
    ~GPUIndexPartition() override;

    void clear() override;

    void reallocate_memory(int64_t new_capacity);

    void append(int64_t n_entry, const idx_t* new_ids, const uint8_t* new_codes);
};
#endif