// C LSTM implementation for dots.tts BigVGAN dec_mi_layer
// Standard LSTM with 4 layers, hidden=512, skip connection
#include <cmath>
#include <cstring>

static float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Single LSTM layer forward (one timestep)
static void lstm_step(
    const float * x,          // [hidden] input
    const float * h_prev,     // [hidden] previous hidden
    const float * c_prev,     // [hidden] previous cell
    const float * w_ih,       // [4*hidden, hidden] input-hidden weights
    const float * w_hh,       // [4*hidden, hidden] hidden-hidden weights
    const float * b_ih,       // [4*hidden] input-hidden bias
    const float * b_hh,       // [4*hidden] hidden-hidden bias
    float * h_out,            // [hidden] output hidden
    float * c_out,            // [hidden] output cell
    int hidden)
{
    // Compute gates: gates = x @ W_ih^T + h @ W_hh^T + b_ih + b_hh
    float gates[2048]; // 4 * hidden (max 2048 for hidden=512)
    for (int i = 0; i < 4*hidden; i++) {
        float g = b_ih[i] + b_hh[i];
        for (int j = 0; j < hidden; j++) {
            g += x[j] * w_ih[i * hidden + j];    // W_ih is [4*hidden, hidden]
            g += h_prev[j] * w_hh[i * hidden + j]; // W_hh is [4*hidden, hidden]
        }
        gates[i] = g;
    }
    
    // Split gates: i, f, g, o
    float * ig = gates;
    float * fg = gates + hidden;
    float * gg = gates + 2*hidden;
    float * og = gates + 3*hidden;
    
    for (int j = 0; j < hidden; j++) {
        float i_val = sigmoid(ig[j]);
        float f_val = sigmoid(fg[j]);
        float g_val = tanhf(gg[j]);
        float o_val = sigmoid(og[j]);
        
        c_out[j] = f_val * c_prev[j] + i_val * g_val;
        h_out[j] = o_val * tanhf(c_out[j]);
    }
}

// 4-layer LSTM with skip connection, batch_first
// x: [batch, seq, hidden] — in our case batch=1
void lstm_forward(
    const float * x,          // [seq * hidden] input
    int seq_len, int hidden, int n_layers,
    const float * w_ih[4],    // per-layer input-hidden weights
    const float * w_hh[4],    // per-layer hidden-hidden weights
    const float * b_ih[4],    // per-layer input-hidden biases
    const float * b_hh[4],    // per-layer hidden-hidden biases
    float * out,              // [seq * hidden] output
    bool skip)                // residual connection
{
    float * h_state = new float[n_layers * hidden](); // hidden states
    float * c_state = new float[n_layers * hidden](); // cell states
    float * tmp_in = new float[hidden];
    float * tmp_out = new float[hidden];
    
    for (int t = 0; t < seq_len; t++) {
        // Copy input for this timestep
        const float * xt = x + t * hidden;
        memcpy(tmp_in, xt, hidden * sizeof(float));
        
        for (int l = 0; l < n_layers; l++) {
            float * h_prev = h_state + l * hidden;
            float * c_prev = c_state + l * hidden;
            
            lstm_step(tmp_in, h_prev, c_prev,
                      w_ih[l], w_hh[l], b_ih[l], b_hh[l],
                      tmp_out, c_prev, hidden);
            
            // Update hidden state
            memcpy(h_prev, tmp_out, hidden * sizeof(float));
            // Output of this layer becomes input to next
            memcpy(tmp_in, tmp_out, hidden * sizeof(float));
        }
        
        // Apply skip connection if enabled
        if (skip) {
            for (int j = 0; j < hidden; j++)
                tmp_out[j] = xt[j] + tmp_out[j];
        }
        
        // Store output
        memcpy(out + t * hidden, tmp_out, hidden * sizeof(float));
    }
    
    delete[] h_state;
    delete[] c_state;
    delete[] tmp_in;
    delete[] tmp_out;
}
