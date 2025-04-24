#include "fp16/Float16.h"
#include "matmul_utils.h"
#include "rknn_api.h"
#include "rknn_matmul_api.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <inttypes.h>

#include <string>
#include <vector>
using namespace rknpu2;
template <typename Ti, typename To>
std::vector<To> matrixMultiply(const Ti *A, const Ti *B, int M, int K, int N)
{
  std::vector<To> result(M * N, 0);

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float sum = 0;
      for (int k = 0; k < K; ++k)
      {
#if DEBUG_PRINT
        printf("A[%d][%d] = %d, B[%d][%d] = %d, A*B = %6d\n", i, k, A[i * K + k], k, j, B[k * N + j],
               A[i * K + k] * B[k * N + j]);
#endif
        sum += (float)A[i * K + k] * (float)B[k * N + j];
      }
      result[i * N + j] = sum;
    }
  }

  return result;
}
std::vector<int8_t> matrixMultiplyWithQuant(const int8_t *A, const int8_t *B, int M, int K, int N, float input_scale,
                                            float weight_scale, float output_scale, int input_zp = 0)
{
  std::vector<int8_t> result(M * N, 0);

  for (int i = 0; i < M; ++i)
  {
    for (int j = 0; j < N; ++j)
    {
      float sum = 0;
      for (int k = 0; k < K; ++k)
      {
        sum += (float)(A[i * K + k] - input_zp) * (float)B[k * N + j];
      }
      result[i * N + j] = sum * input_scale * weight_scale / output_scale;
    }
  }

  return result;
}
static const char *get_dims_string(rknn_matmul_tensor_attr *attr)
{
  if (!attr->n_dims)
  {
    return "()";
  }
  static char dims_str[128];
  memset(&dims_str[0], 0, sizeof(dims_str));
  sprintf(&dims_str[0], "(%d", attr->dims[0]);
  for (uint32_t i = 1; i < attr->n_dims; ++i)
  {
    int idx = strlen(dims_str);
    sprintf(&dims_str[idx], ", %d", attr->dims[i]);
  }
  strcat(&dims_str[0], ")");
  return dims_str;
}
static void dump_matmul_tensor(rknn_tensor_mem *tensor, rknn_matmul_tensor_attr *attr)
{
  printf("  %s%s:\n", attr->name, get_dims_string(attr));
  // normal layout
  if (attr->n_dims == 2)
  {
    for (uint32_t i = 0; i < attr->dims[0]; ++i)
    {
      for (uint32_t j = 0; j < attr->dims[1]; ++j)
      {
        void *virt_addr = (void *)((size_t)tensor->virt_addr + tensor->offset);
        if (attr->type == RKNN_TENSOR_INT8)
        {
          printf(" %4d", ((int8_t *)virt_addr)[i * attr->dims[1] + j]);
        }
        else if (attr->type == RKNN_TENSOR_INT32)
        {
          printf(" %6d", ((int32_t *)virt_addr)[i * attr->dims[1] + j]);
        }
        else if (attr->type == RKNN_TENSOR_FLOAT16)
        {
          printf(" %5.2f", (float)(((float16 *)virt_addr)[i * attr->dims[1] + j]));
        }
        else if (attr->type == RKNN_TENSOR_FLOAT32)
        {
          printf(" %5.2f", ((float *)virt_addr)[i * attr->dims[1] + j]);
        }
        else if (attr->type == RKNN_TENSOR_INT16)
        {
          printf(" %d", ((int16_t *)virt_addr)[i * attr->dims[1] + j]);
        }
      }
      printf("\n");
    }
    printf("\n");
  }
  // perf layout
  else if (attr->n_dims == 3)
  {
    for (uint32_t i = 0; i < attr->dims[0]; ++i)
    {
      for (uint32_t j = 0; j < attr->dims[1]; ++j)
      {
        for (uint32_t k = 0; k < attr->dims[2]; ++k)
        {
          void *virt_addr = (void *)((size_t)tensor->virt_addr + tensor->offset);
          if (attr->type == RKNN_TENSOR_INT8)
          {
            printf(" %4d ", ((int8_t *)virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
          }
          else if (attr->type == RKNN_TENSOR_INT16)
          {
            printf(" %6d ", ((int16_t *)virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
          }
          else if (attr->type == RKNN_TENSOR_INT32)
          {
            printf(" %6d ", ((int32_t *)virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
          }
          else if (attr->type == RKNN_TENSOR_FLOAT16)
          {
            printf(" %5.2f ", (float)(((float16 *)virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]));
          }
          else if (attr->type == RKNN_TENSOR_FLOAT32)
          {
            printf(" %5.2f ", ((float *)virt_addr)[(i * attr->dims[1] + j) * attr->dims[2] + k]);
          }
        }
        printf("\n");
      }
      printf("\n");
    }
  }
  // native layout
  else if (attr->n_dims == 4)
  {
    // N / 16
    for (uint32_t n = 0; n < attr->dims[0]; ++n)
    {
      // K / 32
      for (uint32_t k = 0; k < attr->dims[1]; ++k)
      {
        // 16
        for (uint32_t nn = 0; nn < attr->dims[2]; ++nn)
        {
          // 32
          for (uint32_t kk = 0; kk < attr->dims[3]; kk++)
          {
            void *virt_addr = (void *)((size_t)tensor->virt_addr + tensor->offset);
            if (attr->type == RKNN_TENSOR_INT8)
            {
              printf(" %4d ",
                     ((int8_t *)virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
            }
            else if (attr->type == RKNN_TENSOR_INT32)
            {
              printf(" %6d ",
                     ((int32_t *)virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
            }
            else if (attr->type == RKNN_TENSOR_FLOAT16)
            {
              printf(
                  " %5.2f ",
                  (float)(((float16 *)virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]));
            }
            else if (attr->type == RKNN_TENSOR_FLOAT32)
            {
              printf(" %5.2f ",
                     ((float *)virt_addr)[((n * attr->dims[1] + k) * attr->dims[2] + nn) * attr->dims[3] + kk]);
            }
          }
          printf("\n");
        }
        printf("\n");
      }
      printf("\n");
    }
  }
}
static inline int64_t getCurrentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}
static std::vector<std::string> split(const std::string &str, const std::string &pattern)
{
  std::vector<std::string> res;
  if (str == "")
    return res;
  std::string strs = str + pattern;
  size_t pos = strs.find(pattern);
  while (pos != strs.npos)
  {
    std::string temp = strs.substr(0, pos);
    res.push_back(temp);
    strs = strs.substr(pos + 1, strs.size());
    pos = strs.find(pattern);
  }
  return res;
}

static void print_usage(char *argv[])
{
  printf("Usage:\n%s <matmul_type> <M,K,N> <B_layout> <AC_layout> <loop_count> <core_mask> <print_result> "
         "<iommu_domain_id>\n",
         argv[0]);
  printf("\tmatmul_type = 1: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32\n");
  printf("\tmatmul_type = 2: RKNN_INT8_MM_INT8_TO_INT32\n");
  printf("\tmatmul_type = 4: RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT16\n");
  printf("\tmatmul_type = 7: RKNN_FLOAT16_MM_INT4_TO_FLOAT32\n");
  printf("\tmatmul_type = 10: RKNN_INT4_MM_INT4_TO_INT16\n");
  printf("Example: A = [4,64], B = [64,32], int8 matmul test command as followed:\n");
  printf("%s 2 4,64,32\n", argv[0]);
}

static int pre_process_MKN(std::vector<std::string> MKN_strs, char * argv[]){
    if (MKN_strs.size() != 3)
    {
        fprintf(stderr, "MKN splited by # must be 3 number!\n");
        print_usage(argv);
        return -1;
    }
    return 1;
}

int argv_process(int argc, char * argv[], int &loop_count, int &print_tensor, int &iommu_domain_id, bool &generate_random,
                 int &B_layout, int &AC_layout, int &core_mask){

  printf(">> inside argv_process: %d %d %d\n", loop_count, print_tensor, iommu_domain_id);
  printf(">> inside argv_process: %d %d %d %d\n", generate_random, B_layout, AC_layout, core_mask);
  printf("argv_process\n");
  printf("params: argc = %d\n", argc);
  for(int i = 0; i < argc; i++){
    printf("argv[%d] = %s\n", i, argv[i]);
  }
  if (argc > 3)
  {
    B_layout = atoi(argv[3]);
  }

  // request normal or perf layout for A and C
  if (argc > 4)
  {
    AC_layout = atoi(argv[4]);
  }

  if (argc > 5)
  {
    loop_count = atoi(argv[5]);
  }

  if (argc > 6)
  {
    core_mask = atoi(argv[6]);
  }

  if (argc > 7)
  {
    print_tensor = atoi(argv[7]);
  }

  if (argc > 8)
  {
    iommu_domain_id = atoi(argv[8]);
  }

  if (argc > 9)
  {
      int tmp = atoi(argv[9]);
      generate_random = tmp == 0? false:true;

  }
  printf("return 0\n");
  return 0;

}

rknn_matmul_info init_info(int M, int K, int N, rknn_matmul_type matmul_type, int B_layout, int AC_layout, int iommu_domain_id)
{
    rknn_matmul_info info;
    memset(&info, 0, sizeof(rknn_matmul_info));
    info.M = M;
    info.K = K;
    info.N = N;
    info.type = matmul_type;
    info.B_layout = B_layout;
    info.AC_layout = AC_layout;
    info.iommu_domain_id = iommu_domain_id;

    return info;
}

int rknn_tensor_mem_alloc(rknn_matmul_ctx ctx, rknn_matmul_io_attr *io_attr, rknn_tensor_mem **A, rknn_tensor_mem **B, rknn_tensor_mem **C)
{
    *A = rknn_create_mem(ctx, io_attr->A.size);
    if (*A == NULL)
    {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return -1;
    }
    *B = rknn_create_mem(ctx, io_attr->B.size);
    if (*B == NULL)
    {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return -1;
    }
    *C = rknn_create_mem(ctx, io_attr->C.size);
    if (*C == NULL)
    {
        fprintf(stderr, "rknn_create_mem fail!\n");
        return -1;
    }
    return 0;
}
int main(int argc, char * argv[]){

    printf("start programm\n");
    if (argc < 3)
    {
        print_usage(argv);
        return -1;
    }

    int loop_count = 10;
    int print_tensor = 1;
    int iommu_domain_id = 0;
    bool generate_random = true;

    int B_layout = 0;
    int AC_layout = 0;
    int core_mask = 0;

    rknn_matmul_type matmul_type = (rknn_matmul_type)atoi(argv[1]);
    // check matmul_type value range
    if (matmul_type < 1 || matmul_type > 11)
    {
        fprintf(stderr, "invalid matmul_type: %d, required matmul_type =1~11!\n", matmul_type);
        print_usage(argv);
        return -1;
    }

    std::vector<std::string> MKN_strs = split(argv[2], ",");
    if (pre_process_MKN(MKN_strs, argv) < 0)
        {
            return -1;
        }
    
    int32_t M = std::atoi(MKN_strs[0].c_str());
    int32_t K = std::atoi(MKN_strs[1].c_str());
    int32_t N = std::atoi(MKN_strs[2].c_str());

    {
      if (argc > 3)
      {
        B_layout = atoi(argv[3]);
      }
      // request normal or perf layout for A and C
      if (argc > 4)
      {
        AC_layout = atoi(argv[4]);
      }

      if (argc > 5)
      {
        loop_count = atoi(argv[5]);
      }

      if (argc > 6)
      {
        core_mask = atoi(argv[6]);
      }

      if (argc > 7)
      {
        print_tensor = atoi(argv[7]);
      }

      if (argc > 8)
      {
        iommu_domain_id = atoi(argv[8]);
      }

      if (argc > 9)
      {
          int tmp = atoi(argv[9]);
          generate_random = tmp == 0? false:true;
      }
    }

    printf("params: M = %d, K = %d, N = %d, B_layout = %d, AC_layout = %d, loop_count = %d, core_mask = "
           "%d, iommu_domain_id = %u, generate_random = %d\n",
           M, K, N, B_layout, AC_layout, loop_count, core_mask, iommu_domain_id, generate_random);
    rknn_matmul_ctx ctx;

    /*
        ////////////////////////////////////////////////////////////////////////
        * Step 1:
        * create matmul info and io_attr
        ////////////////////////////////////////////////////////////////////////
    */

    rknn_matmul_info info = init_info(M, K, N, matmul_type, B_layout, AC_layout, iommu_domain_id);

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
    // rknn_matmul_ctx ctx = 0;
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if( ret < 0)
    {
        fprintf(stderr, "rknn_matmul_create fail! ret=%d\n", ret);
        return -1;
    }
    /*
        ////////////////////////////////////////////////////////////////////////
        * Step 2:
        * set core mask
        ////////////////////////////////////////////////////////////////////////
    */

    rknn_matmul_set_core_mask(ctx, (rknn_core_mask)core_mask);
    /*
        ////////////////////////////////////////////////////////////////////////
        * Step 3:
        * create A, B, C float data
        ////////////////////////////////////////////////////////////////////////
    */

    for(int i = 0 ; i < 2 ; i++){
      printf("cnt: %d\n", i);
      void *A_Matrix = nullptr;
      void *B_Matrix = nullptr;
      void *C_Matrix = nullptr;

      if(info.type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32){
          A_Matrix = malloc(M * K * sizeof(float16));
          B_Matrix = malloc(K * N * sizeof(float16));
          C_Matrix = malloc(M * N * sizeof(float));
          float16 *A_float16_Matrix = (float16 *)A_Matrix;
          float16 *B_float16_Matrix = (float16 *)B_Matrix;
          if(generate_random){
              generate_random_buffer(A_float16_Matrix, M * K, {-1.f, 1.f});
              generate_random_buffer(B_float16_Matrix, K * N, {-1.f, 1.f});
          }else{
              generate_fixed_buffer(A_float16_Matrix, M * K, {-1.f, 1.f});
              generate_fixed_buffer(B_float16_Matrix, K * N, {-1.f, 1.f});
          }
      }
      
      /*
          ////////////////////////////////////////////////////////////////////////
          * Step 4:
          * create A, B, C rknn_tensor mem
          ////////////////////////////////////////////////////////////////////////
      */
      rknn_tensor_mem *A=NULL ;
      rknn_tensor_mem *B=NULL ;
      rknn_tensor_mem *C=NULL ;

      if(rknn_tensor_mem_alloc(ctx, &io_attr, &A, &B, &C) < 0)
      {
          fprintf(stderr, "rknn_tensor_mem_alloc fail!\n");
          return -1;
      }

      /*
          ////////////////////////////////////////////////////////////////////////
          * Step 5:
          * data cpy to A, B
          * Assume A and B both use normal layout
          ////////////////////////////////////////////////////////////////////////
      */

      memcpy(A->virt_addr, A_Matrix, M * K * sizeof(float16));
      memcpy(B->virt_addr, B_Matrix, K * N * sizeof(float16));

      /*
          ////////////////////////////////////////////////////////////////////////
          * Step 6:
          * set input and output mem
          ////////////////////////////////////////////////////////////////////////
      */

      printf("step 6\n");
      ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
      if (ret < 0)
      {
          fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
          return -1;
      }
      ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
      if (ret < 0)
      {
          fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
          return -1;
      }
      ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);
      if (ret < 0)
      {
          fprintf(stderr, "rknn_matmul_set_io_mem fail! ret=%d\n", ret);
          return -1;
      }

      /*
          ////////////////////////////////////////////////////////////////////////
          * Step 7:
          * run matmul
          ////////////////////////////////////////////////////////////////////////
      */

      printf("step 7");
      int64_t total_us = 0;
        for (int i = 0; i < loop_count; ++i)
        {
          int64_t start_us = getCurrentTimeUs();
          ret = rknn_matmul_run(ctx);
          int64_t elapse_us = getCurrentTimeUs() - start_us;
          total_us += elapse_us;
          if (ret < 0)
          {
            printf("rknn_matmul_run error %d\n", ret);
            return -1;
          }
          printf("%4d: Elapse Time = %.2fms, FPS = %.2f\n", i, elapse_us / 1000.f, 1000.f * 1000.f / elapse_us);
        }

        /* 
          ////////////////////////////////////////////////////////////////////////
          * Step 8:
          * get result
          ////////////////////////////////////////////////////////////////////////
        */
      printf("step 8");
      if(print_tensor){
          dump_matmul_tensor(A, &io_attr.A);
          dump_matmul_tensor(B, &io_attr.B);
          printf("result:\n");
          dump_matmul_tensor(C, &io_attr.C);
      }
      float *npu_res_ptr = (float *)C->virt_addr;
      if (info.AC_layout == 1)
      {
        int32_t N_remain = io_attr.C.dims[0];
        int32_t subN = io_attr.C.dims[2];
        perf_layout_to_norm_layout(npu_res_ptr, (float *)C_Matrix, M, N, N_remain, subN);
        npu_res_ptr = (float *)C_Matrix;
      }
      std::vector<float> npu_res(npu_res_ptr, npu_res_ptr + M * N);

      // calculate cpu res
      std::vector<float> cpu_res;
      cpu_res.reserve(M * N);
      cpu_res = matrixMultiply<float16, float>((const float16 *)A_Matrix, (const float16 *)B_Matrix, M, K, N);

      if (arraysCosineSimilarity<float>(cpu_res, npu_res))
      {
        printf(
            "FLOAT16_MM_FLOAT16_TO_FLOAT32 matmul result is correct M x K x N is %d %d %d AC_layout is %d B_layout is %d\n",
            M, K, N, AC_layout, B_layout);
        ret = 0;
      }
      else
      {
        printf(
            "FLOAT16_MM_FLOAT16_TO_FLOAT32 matmul result is wrong M x K x N is %d %d %d AC_layout is %d B_layout is %d\n",
            M, K, N, AC_layout, B_layout);
        ret = -1;
      }

      if (A_Matrix)
      {
        free(A_Matrix);
      }
      if (B_Matrix)
      {
        free(B_Matrix);
      }

    }
      // rknn_destroy_mem(ctx, A);
      // rknn_destroy_mem(ctx, B);
      // rknn_destroy_mem(ctx, C);

      rknn_matmul_destroy(ctx);


    return 0;
}
