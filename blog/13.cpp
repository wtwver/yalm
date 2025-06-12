float sum = 0.0;
for (int t = warp_id; t < seq_len; t += t_stride) {
  sum += vh[kv_stride * t + i] * atth[t];	
}
// atomicAdd `sum` when we're done
      

float sum = 0.0;
for (int t = warp_id; t < seq_len; t += t_stride) {
  sum += __half2float(vh[kv_stride * t + i]) * atth[t];	
}
// atomicAdd `sum` when we're done

float2 sum01 = make_float2(0.0, 0.0);
for (int t = warp_id; t < seq_len; t += t_stride) {
  float2 v01 = __half22float2(*((half2*)&vh[kv_stride * t + i]));
  float att_t = atth[t];
  // Sadly CUDA does not have float2 SIMD ops
  sum01.x += v01.x * att_t;
  sum01.y += v01.y * att_t;
}
// atomicAdd both `sum01` lanes when we're done