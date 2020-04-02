#ifndef GPUDATADOTH
#define GPUDATADOTH

#define RTERROR(status, s) \
    if (status != cudaSuccess) { \
        printf("%s %s\n", s, cudaGetErrorString(status)); \
        assert(0); \
        cudaDeviceReset(); \
        exit(-1); \
    }
    
    
#ifdef SYNCHRONOUS
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            cudaDeviceReset(); \
            exit(-1); \
        } \
        cudaDeviceSynchronize(); \
    }
#else
#define LAUNCHERROR(s) \
    { \
        cudaError_t status = cudaGetLastError(); \
        if (status != cudaSuccess) { \
            printf("Error: %s launching kernel %s\n", cudaGetErrorString(status), s); \
            cudaDeviceReset(); \
            exit(-1); \
        } \
    }
#endif

typedef struct
{
        int             num_of_atoms;
        int             num_of_atypes;
        int             num_of_intraE_contributors;
        int             gridsize_x;
        int             gridsize_y;
        int             gridsize_z;
        int             gridsize_x_times_y;
        int             gridsize_x_times_y_times_z;
        float           grid_spacing;
        int             rotbondlist_length;
        float           coeff_elec;
        float           coeff_desolv;
        int*            evals_of_new_entities;
        unsigned int*   prng_states;
        int             pop_size;
        int             num_of_genes;
        float           tournament_rate;
        float           crossover_rate;
        float           mutation_rate;
        float           abs_max_dmov;
        float           abs_max_dang;
        float           lsearch_rate;
        float           smooth;
        unsigned int    num_of_lsentities;
        float           rho_lower_bound;
        float           base_dmov_mul_sqrt3;
        float           base_dang_mul_sqrt3;
        unsigned int    cons_limit;
        unsigned int    max_num_of_iters;
        float           qasp;
} GpuDockparameters;

struct GpuData {
    GpuDockparameters               dockpars;
    
    // Consolidated constants and memory pointers to reduce kernel launch overhead
    kernelconstant_interintra*      pKerconst_interintra;
	kernelconstant_intracontrib*    pKerconst_intracontrib;
	kernelconstant_intra*	        pKerconst_intra;
	kernelconstant_rotlist*		    pKerconst_rotlist;
	kernelconstant_conform*		    pKerconst_conform;
	kernelconstant_grads*		    pKerconst_grads;
    float*                          pMem_fgrids;
    int*                            pMem_evals_of_new_entities;
    int*                            pMem_gpu_evals_of_runs;
    uint32_t*                       pMem_prng_states;
    int                             mem_rotbonds_const[2*MAX_NUM_OF_ROTBONDS];
    int                             mem_rotbonds_atoms_const[MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS];
    int                             mem_num_rotating_atoms_per_rotbond_const[MAX_NUM_OF_ROTBONDS];
    float                           mem_angle_const[1000];
    float                           mem_dependence_on_theta_const[1000];
    float                           mem_dependence_on_rotangle_const[1000];
    
    // CUDA-specific constants
    unsigned int                    warpmask;
    unsigned int                    warpbits;
};
#endif


