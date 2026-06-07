// dots.tts.cpp - Flow Matching ODE solver
// Solves dz/dt = v_theta(z_t, t | conditioning)
// Starting from z_0 ~ N(0, I), ending at z_1 (generated latent patch)

#include "dots_tts.h"
#include "dots_tts_util.h"
#include "dit.h"
#include "ggml.h"
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <cstdio>

// ===========================================================================
// Flow Matching ODE Solvers
// ===========================================================================
// The velocity field v_theta predicts dz/dt.
// We solve from t=0 (noise) to t=1 (data) by integrating v_theta.
//
// For flow matching with optimal transport path:
//   z_t = (1-t)*z_0 + t*z_1
//   dz/dt = z_1 - z_0 = v_theta(z_t, t | cond)
//
// So z_1 = z_0 + integral_0^1 v_theta(z_t, t) dt
// ---------------------------------------------------------------------------

struct fm_solver_ctx {
    // Components for velocity field evaluation
    dit_model * dit;
    ggml_context * w_ctx;    // weight context (persistent)
    ggml_context * compute_ctx; // graph context (per-step, reset each call)
    ggml_cgraph * gf;

    // Conditioning buffers (pre-computed per FM step)
    float * cond_hidden;      // [seq_len, hidden] — projected LLM hiddens + history
    float * speaker_emb;      // [speaker_dim] — x-vector
    int cond_seq_len;
    int cond_hidden_dim;
    int latent_dim;
    int latent_patch_size;    // 4 (VAE patch_size)
    int nfe;                  // number of function evaluations for ODE
    bool use_cfg;             // classifier-free guidance
    float cfg_scale;          // guidance strength

    // Temporary buffers for noise
    float * z_t;              // [latent_patch_size * latent_dim] current state
    float * z_out;            // [latent_patch_size * latent_dim] output

    // ggml backend - CPU only for MVP
    // s->backend = ...  (not used yet)
};

// Initialize solver
fm_solver_ctx * fm_solver_init(
    dit_model * dit,
    ggml_context * w_ctx,
    int cond_seq_len,
    int cond_hidden_dim,
    float * speaker_emb,
    int nfe, bool use_cfg, float cfg_scale
) {
    fm_solver_ctx * s = new fm_solver_ctx();
    s->dit = dit;
    s->w_ctx = w_ctx;
    s->cond_seq_len = cond_seq_len;
    s->cond_hidden_dim = cond_hidden_dim;
    s->latent_dim = VAE_LATENT_DIM;
    s->latent_patch_size = PATCHENC_PATCH_SIZE;
    s->nfe = nfe;
    s->use_cfg = use_cfg;
    s->cfg_scale = cfg_scale;

    int z_size = s->latent_patch_size * s->latent_dim;

    // Speaker embedding
    s->speaker_emb = new float[DIT_SPEAKER_DIM]();
    if (speaker_emb) {
        memcpy(s->speaker_emb, speaker_emb, DIT_SPEAKER_DIM * sizeof(float));
    }

    // Conditioning buffer
    s->cond_hidden = new float[cond_seq_len * cond_hidden_dim]();

    s->z_t   = new float[z_size]();
    s->z_out = new float[z_size]();

    // Compute context (re-created each step)
    struct ggml_init_params compute_params = {
        /*.mem_size   =*/ 512ULL * 1024 * 1024,  // 512 MB
        /*.mem_buffer =*/ nullptr,
        /*.no_alloc   =*/ false,
    };
    s->compute_ctx = ggml_init(compute_params);

    // Backend (CPU for MVP)
    // s->backend = ggml_backend_reg_get_default()->init()

    return s;
}

void fm_solver_free(fm_solver_ctx * s) {
    delete[] s->cond_hidden;
    delete[] s->speaker_emb;
    delete[] s->z_t;
    delete[] s->z_out;
    ggml_free(s->compute_ctx);
    delete s;
}

// Set conditioning for current FM step
void fm_solver_set_cond(fm_solver_ctx * s, float * hidden, int seq_len, int hidden_dim) {
    memcpy(s->cond_hidden, hidden, seq_len * hidden_dim * sizeof(float));
    s->cond_seq_len = seq_len;
}

// Evaluate velocity field v_theta(z_t, t | cond) with optional CFG
static void fm_solver_velocity(
    fm_solver_ctx * s,
    float * z_t,      // [latent_patch_size, latent_dim]
    float t,           // scalar timestep in [0, 1]
    float * v_out,     // [latent_patch_size, latent_dim] velocity
    bool unconditional // if true, run without speaker conditioning (for CFG)
) {
    ggml_context * ctx = s->compute_ctx;
    ggml_reset(ctx);

    int seq_len = s->cond_seq_len;
    int hidden  = s->cond_hidden_dim;
    int latent  = s->latent_dim;
    int patch   = s->latent_patch_size;
    int n_batch = 1;
    int total_seq = seq_len + patch;  // conditioning + noise patches

    // Build input tensor: [total_seq, 1, hidden]
    // First seq_len positions: projected LLM hiddens + history latents (conditioning)
    // Last patch positions: noise z_t projected to hidden_dim
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, hidden, n_batch, total_seq);

    float * x_data = tensor_data(x);
    // Copy conditioning
    memcpy(x_data, s->cond_hidden, seq_len * hidden * sizeof(float));
    // Project noise to hidden (simple linear — will need noise_in_proj later)
    // For now, pad noise with zeros to hidden_dim
    float * noise_slot = x_data + seq_len * hidden;
    for (int i = 0; i < patch; i++) {
        for (int j = 0; j < latent && j < hidden; j++) {
            noise_slot[i * hidden + j] = z_t[i * latent + j];
        }
    }

    // Reshape to [total_seq, n_batch, hidden]
    x = ggml_cont(ctx, ggml_permute(ctx, x, 2, 1, 0, 3));

    // Timestep tensor
    ggml_tensor * t_tensor = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ((float *)t_tensor->data)[0] = t;

    // Speaker embedding (or zero for unconditional)
    ggml_tensor * spk = nullptr;
    if (!unconditional && s->speaker_emb) {
        spk = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, DIT_SPEAKER_DIM);
        memcpy(spk->data, s->speaker_emb, DIT_SPEAKER_DIM * sizeof(float));
        // reshape to [speaker_dim, 1]
        spk = ggml_reshape_2d(ctx, spk, DIT_SPEAKER_DIM, 1);
    }

    // Run DiT forward
    ggml_tensor * v = dit_forward(*s->dit, ctx, x, t_tensor, spk);

    // Extract last patch_size elements (the generated velocity)
    // v is [total_seq, latent_dim] — we want last patch rows
    // Actually dit_forward returns [latent_patch_size, latent_dim]
    // Copy to output
    int out_size = patch * latent;
    memcpy(v_out, v->data, out_size * sizeof(float));

    ggml_reset(ctx);
}

// ===========================================================================
// Euler solver (fixed-step)
// ===========================================================================

void fm_solve_euler(fm_solver_ctx * s, float * z_1) {
    int z_size = s->latent_patch_size * s->latent_dim;
    int steps = s->nfe;
    float dt = 1.0f / (float)steps;

    // z_0 = random noise
    for (int i = 0; i < z_size; i++) {
        // Box-Muller for Gaussian
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        if (u1 < 1e-6f) u1 = 1e-6f;
        s->z_t[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
    }

    // Euler integration: z_{t+dt} = z_t + v_theta(z_t, t) * dt
    for (int step = 0; step < steps; step++) {
        float t = (float)step * dt;

        float v[z_size];
        if (s->use_cfg) {
            // CFG: v = v_uncond + cfg_scale * (v_cond - v_uncond)
            float v_cond[z_size];
            float v_uncond[z_size];
            fm_solver_velocity(s, s->z_t, t, v_cond, false);
            fm_solver_velocity(s, s->z_t, t, v_uncond, true);
            for (int i = 0; i < z_size; i++) {
                v[i] = v_uncond[i] + s->cfg_scale * (v_cond[i] - v_uncond[i]);
            }
        } else {
            fm_solver_velocity(s, s->z_t, t, v, false);
        }

        // z_{t+dt} = z_t + v * dt
        for (int i = 0; i < z_size; i++) {
            s->z_t[i] += v[i] * dt;
        }
    }

    memcpy(z_1, s->z_t, z_size * sizeof(float));
}

// ===========================================================================
// Midpoint solver (RK2)
// ===========================================================================

void fm_solve_midpoint(fm_solver_ctx * s, float * z_1) {
    int z_size = s->latent_patch_size * s->latent_dim;
    int steps = s->nfe;
    float dt = 1.0f / (float)steps;

    // Initialize with Gaussian noise
    for (int i = 0; i < z_size; i++) {
        float u1 = (float)rand() / (float)RAND_MAX;
        float u2 = (float)rand() / (float)RAND_MAX;
        if (u1 < 1e-6f) u1 = 1e-6f;
        s->z_t[i] = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.1415926535f * u2);
    }

    auto velocity = [s](float * z, float t, float * v_out) {
        if (s->use_cfg) {
            int sz = s->latent_patch_size * s->latent_dim;
            float v_cond[sz], v_uncond[sz];
            fm_solver_velocity(s, z, t, v_cond, false);
            fm_solver_velocity(s, z, t, v_uncond, true);
            for (int i = 0; i < sz; i++) {
                v_out[i] = v_uncond[i] + s->cfg_scale * (v_cond[i] - v_uncond[i]);
            }
        } else {
            fm_solver_velocity(s, z, t, v_out, false);
        }
    };

    for (int step = 0; step < steps; step++) {
        float t = (float)step * dt;

        // k1 = v(z_t, t)
        float k1[z_size];
        float z_mid[z_size];
        velocity(s->z_t, t, k1);

        // z_mid = z_t + k1 * dt/2
        for (int i = 0; i < z_size; i++) {
            z_mid[i] = s->z_t[i] + k1[i] * dt * 0.5f;
        }

        // k2 = v(z_mid, t + dt/2)
        float k2[z_size];
        velocity(z_mid, t + dt * 0.5f, k2);

        // z_{t+dt} = z_t + k2 * dt
        for (int i = 0; i < z_size; i++) {
            s->z_t[i] += k2[i] * dt;
        }
    }

    memcpy(z_1, s->z_t, z_size * sizeof(float));
}
