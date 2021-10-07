

#ifndef AUXILIARY_GENETIC_H  
#define AUXILIARY_GENETIC_H  

#pragma omp declare target
 uint32_t gpu_rand(
                uint32_t* prng_states,
                int blockIdx, int threadIdx
);

 float gpu_randf(
                uint32_t* prng_states,
                int blockIdx, int threadIdx
);

 void map_angle(
		float& angle
);
#pragma omp end declare target

#endif
