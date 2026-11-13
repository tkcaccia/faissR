#ifndef FAISSR_GPU_KNN_HANDLE_HPP
#define FAISSR_GPU_KNN_HANDLE_HPP

// Shared owner layout for native CUDA and FAISS GPU result buffers.
struct FaissRGpuKnnHandle {
  int* indices = nullptr;
  float* distances = nullptr;
  int n_points = 0;
  int k = 0;
  int device = 0;
};

#endif
