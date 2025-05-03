#include "matmul_utils.h"
#include "fp16/Float16.h"
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

bool is_same_matrix(const void *src, const void *dst, int32_t M, int32_t K)
{
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < K; j++)
        {
            if (((float16 *)src)[i * K + j] != ((float16 *)dst)[i * K + j])
            {
                printf("src[%d][%d] = %f, dst[%d][%d] = %f\n", i, j, ((float16 *)src)[i * K + j], i, j,
                       ((float16 *)dst)[i * K + j]);
                return false;
            }
        }
    }
    return true;
}
void transposed_matrix_to_perf_layout(const void *src, void *dst, int32_t K, int32_t N, int32_t subK, int32_t subN)
{
    // subN = 16 subK = 32 on 3588
    const void * start_point = src;
    int dst_offset = 0;
    for(int outer = 0; outer < N / subN; outer++){
        for(int j = 0 ; j < K / subK; j++){
            int start_offset = outer * subN * K + j * subK;
            for(int i = 0 ; i < subN; i++){
                memcpy((float16 *)dst + dst_offset, (float16 *)start_point + start_offset, subK * sizeof(float16));
                dst_offset += subK;
                start_offset += K;
            }
        }
    }
}

void perf_matrixC_to_norm_layout(const void *src, void *dst, int32_t M, int32_t N){
    const int mem_unit = 8;
    int dst_offset = 0;
    for(int outer = 0; outer < N / mem_unit ; outer++){
        for(int i = 0; i < M; i++){
            int src_offset = outer * mem_unit + i *N;
            memcpy((float16 *)dst + dst_offset, (float16 *)src + src_offset, mem_unit * sizeof(float16));
            dst_offset += mem_unit;
        }
    }
}
void matrixA_to_perf_layout(const void* src, void *dst, int32_t M, int32_t K){
    const int mem_unit = 8;
    int dst_offset = 0;
    for(int outer = 0; outer < K / mem_unit ; outer++){
        for(int i = 0; i < M; i++){
            int src_offset = outer * mem_unit + i *K;
            memcpy((float16 *)dst + dst_offset, (float16 *)src + src_offset, mem_unit * sizeof(float16));
            dst_offset += mem_unit;
        }
    }
}
void transpose_matrix(const void * src, void *dst, int32_t K, int32_t N){
    for (int32_t k = 0; k < K; ++k)
    {
        for (int32_t n = 0; n < N; ++n)
        {
            ((float16 *)dst)[n * K + k] = ((float16 *)src)[k * N + n];
        }
    }
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
int main(int argc, char *argv[])
{
    if (argc < 3)
    {
        print_usage(argv);
        return -1;
    }

    // The quantization parameters used in this demo are set for test and the actual scale and zp are based on customer usage scenarios.
    // all scales and zp are the same as those in B per-channel quantization, and the result is the same as the result of per-layer quantization.
    const float A_SCALE = 0.2f;
    const float B_SCALE = 0.1f;
    const float C_SCALE = 0.8f;
    const int A_ZP = 0;
    const int B_ZP = 0;
    const int C_ZP = 0;

    int loop_count = 10;
    int print_tensor = 1;
    int iommu_domain_id = 0;
    bool generate_random = true;

    rknn_matmul_type matmul_type = (rknn_matmul_type)atoi(argv[1]);
    // check matmul_type value range
    if (matmul_type < 1 || matmul_type > 11)
    {
        fprintf(stderr, "invalid matmul_type: %d, required matmul_type =1~11!\n", matmul_type);
        print_usage(argv);
        return -1;
    }

    std::vector<std::string> MKN_strs = split(argv[2], ",");
    if (MKN_strs.size() != 3)
    {
        fprintf(stderr, "MKN splited by # must be 3 number!\n");
        print_usage(argv);
        return -1;
    }
    int32_t M = std::atoi(MKN_strs[0].c_str());
    int32_t K = std::atoi(MKN_strs[1].c_str());
    int32_t N = std::atoi(MKN_strs[2].c_str());

    // request normal or native layout for B
    int B_layout = 0;
    if (argc > 3)
    {
        B_layout = atoi(argv[3]);
    }

    // request normal or perf layout for A and C
    int AC_layout = 0;
    if (argc > 4)
    {
        AC_layout = atoi(argv[4]);
    }

    if (argc > 5)
    {
        loop_count = atoi(argv[5]);
    }

    int core_mask = 0;
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
        generate_random = tmp == 0 ? false : true;
    }
    printf("params: M = %d, K = %d, N = %d, B_layout = %d, AC_layout = %d, loop_count = %d, core_mask = "
        "%d, iommu_domain_id = %u, generate_random = %d\n",
        M, K, N, B_layout, AC_layout, loop_count, core_mask, iommu_domain_id, generate_random);


    rknn_matmul_ctx ctx;

    rknn_matmul_info info = init_info(M, K, N, matmul_type, B_layout, AC_layout, iommu_domain_id);

    rknn_matmul_io_attr io_attr;
    memset(&io_attr, 0, sizeof(rknn_matmul_io_attr));
    // rknn_matmul_ctx ctx = 0;
    int ret = rknn_matmul_create(&ctx, &info, &io_attr);
    if (ret < 0)
    {
        fprintf(stderr, "rknn_matmul_create fail! ret=%d\n", ret);
        return -1;
    }

    rknn_matmul_set_core_mask(ctx, (rknn_core_mask)core_mask);

    void *A_Matrix = nullptr;
    void *B_Matrix = nullptr;
    void *C_Matrix = nullptr;

    if (info.type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32)
    {
        A_Matrix = malloc(M * K * sizeof(float16));
        B_Matrix = malloc(K * N * sizeof(float16));
        C_Matrix = malloc(M * N * sizeof(float));
        float16 *A_float16_Matrix = (float16 *)A_Matrix;
        float16 *B_float16_Matrix = (float16 *)B_Matrix;
        if (generate_random)
        {
            generate_random_buffer(A_float16_Matrix, M * K, {-1.f, 1.f});
            generate_random_buffer(B_float16_Matrix, K * N, {-1.f, 1.f});
        }
        else
        {
            generate_fixed_buffer(A_float16_Matrix, M * K, {-1.f, 1.f});
            generate_fixed_buffer(B_float16_Matrix, K * N, {-1.f, 1.f});
        }
    }

    // Check generated matrix


      rknn_tensor_mem *A=NULL ;
      rknn_tensor_mem *B=NULL ;
      rknn_tensor_mem *C=NULL ;

      if(rknn_tensor_mem_alloc(ctx, &io_attr, &A, &B, &C) < 0)
      {
          fprintf(stderr, "rknn_tensor_mem_alloc fail!\n");
          return -1;
      }


    int32_t subK = io_attr.A.dims[2];
    norm_layout_to_perf_layout<float16, float16>(static_cast<float16 *>(A_Matrix), static_cast<float16 *>(A->virt_addr), M, K, subK, false);

    //check how perf_layout is
    void * native_A_Matrix = malloc(M * K * sizeof(float16));

    matrixA_to_perf_layout(A_Matrix, native_A_Matrix, M, K);

    bool same_matrix = is_same_matrix(A->virt_addr, native_A_Matrix, M, K);
    if(same_matrix)
        printf("Matrix A Same!\n");
    else
        printf("Matrix A Wrong!\n");



    int32_t B_subN = io_attr.B.dims[2];
    int32_t B_subK = io_attr.B.dims[3];
    norm_layout_to_native_layout(static_cast<float16 *>(B_Matrix), static_cast<float16 *>(B->virt_addr), K, N, B_subN, B_subK, false);
    // rknn_B_normal_layout_to_native_layout(B_Matrix, B->virt_addr, K, N, &info);
    //check how native layout is
    void * transpose_B_Matrix = malloc(K * N * sizeof(float16));
    transpose_matrix(B_Matrix, transpose_B_Matrix, K, N);
    void * native_B_Matrix = malloc(K * N * sizeof(float16));
    transposed_matrix_to_perf_layout(transpose_B_Matrix, native_B_Matrix, K, N, B_subK, B_subN);


    // check if B->virt_addr is the same as native B

    bool checked = is_same_matrix(B->virt_addr, native_B_Matrix, N, K);
    if(checked)
        printf("Matrix B Same!\n");
    else
        printf("Matrix B Wrong!\n");

    
    memcpy(A->virt_addr, native_A_Matrix, M * K * sizeof(float16));
    memcpy(B->virt_addr, native_B_Matrix, K * N * sizeof(float16));

    int ret = rknn_matmul_set_io_mem(ctx, A, &io_attr.A);
    ret = rknn_matmul_set_io_mem(ctx, B, &io_attr.B);
    ret = rknn_matmul_set_io_mem(ctx, C, &io_attr.C);

    ret = rknn_matmul_run(ctx);
    if(ret < 0){
        printf("rknn_matmul run error %d\n", ret);
        return -1;
    }

    if (info.AC_layout == 1)
    {
        int32_t N_remain = io_attr.C.dims[0];
        int32_t subN = io_attr.C.dims[2];
        perf_layout_to_norm_layout(static_cast<float16 *>(C->virt_addr), static_cast<float16 *>(C_Matrix), M, N, N_remain, subN);
        C_Matrix = C_Matrix;
    }


    return 0;
}
