
cdef extern from "bingham.h":
    ctypedef struct bingham_stats_t:
        double *dF
        double entropy
        double *mode
        double **scatter
   
    ctypedef struct bingham_t:
        int d
        double **V
        double *Z
        double F
        bingham_stats_t *stats

    ctypedef struct bingham_pmf_t:
        pass
    ctypedef struct bingham_mix_t:
        pass

    void bingham_init()
    void bingham_new(bingham_t *B, int d, double **V, double *Z)
    void bingham_new_uniform(bingham_t *B, int d)
    void bingham_set_uniform(bingham_t *B)
    void bingham_new_S1(bingham_t *B, double *v1, double z1)
    void bingham_new_S2(bingham_t *B, double *v1, double *v2, double z1, double z2)
    void bingham_new_S3(bingham_t *B, double *v1, double *v2, double *v3, double z1, double z2, double z3)
    void bingham_copy(bingham_t *dst, bingham_t *src)
    void bingham_alloc(bingham_t *B, int d)
    void bingham_free(bingham_t *B)
    double bingham_F(bingham_t *B)
    double bingham_pdf(double x[], bingham_t *B)
    double bingham_L(bingham_t *B, double **X, int n)
    int bingham_is_uniform(bingham_t *B)
    void bingham_mode(double *mode, bingham_t *B)
    void bingham_stats(bingham_t *B)
    void bingham_free_stats(bingham_t *B)
    void bingham_stats_free(bingham_stats_t *stats)
    double bingham_cross_entropy(bingham_t *B1, bingham_t *B2)
    double bingham_KL_divergence(bingham_t *B1, bingham_t *B2)
    void bingham_merge(bingham_t *B, bingham_t *B1, bingham_t *B2, double alpha)
    void bingham_compose(bingham_t *B, bingham_t *B1, bingham_t *B2)
    double bingham_compose_true_pdf(double *x, bingham_t *B1, bingham_t *B2)
    double bingham_compose_error(bingham_t *B1, bingham_t *B2)
    void bingham_fit(bingham_t *B, double **X, int n, int d)
    void bingham_fit_scatter(bingham_t *B, double **S, int d)
    void bingham_discretize(bingham_pmf_t *pmf, bingham_t *B, int ncells)
    void bingham_sample_uniform(double **X, int d, int n)
    void bingham_sample(double **X, bingham_t *B, int n)
    void bingham_sample_pmf(double **X, bingham_pmf_t *pmf, int n)
    void bingham_sample_ridge(double **X, bingham_t *B, int n, double pthresh)
    void bingham_cluster(bingham_mix_t *BM, double **X, int n, int d)
    void bingham_mult(bingham_t *B, bingham_t *B1, bingham_t *B2)
    void bingham_mult_array(bingham_t *B, bingham_t *B_array, int n, int compute_F)
    void print_bingham(bingham_t *B)

    void bingham_pre_rotate_3d(bingham_t *B_rot, bingham_t *B, double *q); 
    void bingham_post_rotate_3d(bingham_t *B_rot, bingham_t *B, double *q);
    void bingham_invert_3d(bingham_t *B_inv, bingham_t *B);

    ## Bingham mixtures ##
    void bingham_mixture_mult(bingham_mix_t *BM, bingham_mix_t *BM1, bingham_mix_t *BM2)
    void bingham_mixture_copy(bingham_mix_t *dst, bingham_mix_t *src)
    void bingham_mixture_free(bingham_mix_t *BM)
    void bingham_mixture_sample(double **X, bingham_mix_t *BM, int n)
    void bingham_mixture_sample_ridge(double **X, bingham_mix_t *BM, int n, double pthresh)
    double bingham_mixture_pdf(double x[], bingham_mix_t *BM)
    void bingham_mixture_add(bingham_mix_t *dst, bingham_mix_t *src)
    double bingham_mixture_peak(bingham_mix_t *BM)
    void bingham_mixture_thresh_peaks(bingham_mix_t *BM, double pthresh)
    void bingham_mixture_thresh_weights(bingham_mix_t *BM, double wthresh)
    bingham_mix_t *load_bmx(char *f_bmx, int *k)
    void save_bmx(bingham_mix_t *BM, int num_clusters, char *fout) 


cdef extern from "bingham/bingham_constants.h":
    double bingham_F_lookup_3d(double *Z)
    double bingham_F_3d(double z1, double z2, double z3)
    double bingham_dF1_3d(double z1, double z2, double z3)
    double bingham_dF2_3d(double z1, double z2, double z3)
    double bingham_dF3_3d(double z1, double z2, double z3)


cdef extern from "bingham/util.h":
    double **new_matrix2(int n, int m)
