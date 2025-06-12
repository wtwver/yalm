// F16C code technically operates on 16-bit unsigned short integers
typedef uint16_t f16_t;

// matmul supporting float16 weights via the F16C extension, which allows
// conversion into float32 values before calculations.
static void matmul(float* xout, float* x, f16_t* w, int n, int d) {
#if defined(__AVX2__) && defined(__F16C__)
  // W (d,n) @ x (n,) -> xout (d,)
  assert(n % 16 == 0);
  int i;
#pragma omp parallel for private(i)
  for (i = 0; i < d; i++) {
    // Vectorized dot product of w[i][:] and x[:] where w is a packed float16 array.
    __m256 sumlo = _mm256_setzero_ps();
    __m256 sumhi = _mm256_setzero_ps();
    for (int j = 0; j < n; j+=16) {
      // Extract the next set of 16 float16 weights from `w` and store them
      // to two separate float32 vectors of width 8 (`wveclo_ps`, `wvechi_ps`)
      __m256i wvec = _mm256_loadu_si256((__m256i*)&w[i * n + j]);
      __m128i wveclo = _mm256_extractf128_si256(wvec, 0);
      __m128i wvechi = _mm256_extractf128_si256(wvec, 1);
      __m256 wveclo_ps = _mm256_cvtph_ps(wveclo);
      __m256 wvechi_ps = _mm256_cvtph_ps(wvechi);
      // Extract the next two float32 vectors of width 8 `xveclo`, `xvechi` from `x`
      __m256 xveclo = _mm256_loadu_ps(&x[j]);
      __m256 xvechi = _mm256_loadu_ps(&x[j + 8]);
      // Compute vectorized FMAs: sumlo += wveclo * xveclo, sumhi += wvechi * xvechi
      sumlo = _mm256_fmadd_ps(wveclo_ps, xveclo, sumlo);
      sumhi = _mm256_fmadd_ps(wvechi_ps, xvechi, sumhi);
    }
    // Horizontally reduce width-8 float32 vectors sumlo, sumhi to a scalar.
    __m256 sum8 = _mm256_add_ps(sumlo, sumhi);              // sum8[0:8] = sumlo[0:8] + sumhi[0:8]
    __m128 sum4 = _mm_add_ps(                               // sum4[0:4] = sum8[0:4] + sum8[4:8]
      _mm256_extractf128_ps(sum8, 0), 
      _mm256_extractf128_ps(sum8, 1)
    );
    __m128 sum1 = _mm_dp_ps(sum4, _mm_set1_ps(1.0f), 0xf1); // sum1[0] = dot(sum4, [1,1,1,1])
    xout[i] = _mm_cvtss_f32(sum1);
  }
#else
  assert(false && "float16 not supported on this platform");
#endif
}
        