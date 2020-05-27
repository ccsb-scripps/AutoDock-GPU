typedef struct
{
        int             num_of_atoms;
        int             num_of_atypes;
	int		num_of_map_atypes;
        int             num_of_intraE_contributors;
        int             gridsize_x;
        int             gridsize_y;
        int             gridsize_z;
        float           grid_spacing;
        float*          fgrids;
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


typedef struct
{
       float atom_charges_const[MAX_NUM_OF_ATOMS];
       uint32_t  atom_types_const  [MAX_NUM_OF_ATOMS];
       uint32_t  atom_types_map_const[MAX_NUM_OF_ATOMS];
} kernelconstant_interintra;

typedef struct
{
       int  intraE_contributors_const[3*MAX_INTRAE_CONTRIBUTORS];
} kernelconstant_intracontrib;

typedef struct
{
       float reqm_const [2*ATYPE_NUM]; // 1st ATYPE_NUM entries = vdW, 2nd ATYPE_NUM entries = hbond
       unsigned int  atom1_types_reqm_const [ATYPE_NUM];
       unsigned int  atom2_types_reqm_const [ATYPE_NUM];
       float VWpars_AC_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float VWpars_BD_const   [MAX_NUM_OF_ATYPES*MAX_NUM_OF_ATYPES];
       float dspars_S_const    [MAX_NUM_OF_ATYPES];
       float dspars_V_const    [MAX_NUM_OF_ATYPES];
} kernelconstant_intra;

typedef struct
{
       int   rotlist_const     [MAX_NUM_OF_ROTATIONS];
} kernelconstant_rotlist;

typedef struct
{
       float ref_coords_const[3*MAX_NUM_OF_ATOMS];
       float rotbonds_moving_vectors_const[3*MAX_NUM_OF_ROTBONDS];
       float rotbonds_unit_vectors_const  [3*MAX_NUM_OF_ROTBONDS];
       float ref_orientation_quats_const  [4*MAX_NUM_OF_RUNS];
} kernelconstant_conform;


struct GpuData {
    GpuDockparameters               dockpars;
    
    // Consolidated constants and memory pointers to reduce kernel launch overhead
    kernelconstant_interintra*      pKerConst_interintra;
	kernelconstant_intracontrib*    pKerConst_intracontrib;
	kernelconstant_intra*	        pKerConst_intra;
	kernelconstant_rotlist*		    pKerConst_rotlist;
	kernelconstant_conform*		    pKerConst_conform;
	kernelconstant_grads*		    pKerConst_grads;
    float*                          pMem_fgrids;
    int*                            pMem_evals_of_new_entities;
    int*                            pMem_gpu_evals_of_runs;
    int*                            pMem_prng_states;
    int                             mem_rotbonds_const[2*MAX_NUM_OF_ROTBONDS];
    int                             mem_rotbonds_atoms_const[MAX_NUM_OF_ATOMS*MAX_NUM_OF_ROTBONDS];
    int                             mem_num_rotating_atoms_per_rotbond_const[MAX_NUM_OF_ROTBONDS];
    float                           mem_angle_const[1000];
    float                           mem_dependence_on_theta_const[1000];
    float                           mem_dependence_on_rotangle_const[1000];
    
    // CUDA-specific constants
    unsigned int                    warpmask;
};


