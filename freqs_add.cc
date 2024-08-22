#include "freqs_ntT.hh"

#include <cmath>
#include <iostream>

using std::abs;
using std::cout;
using std::endl;

void freqs_ntT::interp_extrap(double e_val, double Tcm, double* all_outputs)
{
    int index_less = -1;
    
    if (e_val < eps->get_min_linspace())
    {
        index_less = int(eps->get_gel() / 2)
        if (e_val < eps->get_value(index_less))
            while (e_val < eps->get_value(index_less))
            {
                index_less--;
                if (index_less == 0)
                    break;
            }
        else
        {
            while (e_val > eps->get_value(index_less))
            {
                index_less++;
                if (index_less == eps->get_len())
                    break;
            }
            index_less--;
        }
    }
    else if (e_val > eps->get_value(eps->get_len()-1))
        index_less = -1;
    else
    {
        double delta_eps_est = eps->get_value(eps->get_gel()+2) - eps->get_value(eps->get_gel()+1);
        index_less = min( eps->get_len()-1, int(e_val/delta_eps_est + eps->get_gel()));
        
        if (e_val < eps->get_value(index_less))
            while (e_val < eps->get_value(index_less))
                index_less--;
        else
        {
            while (e_val > eps->get_value(index_less))
            {
                index_less++;
                if (index_less == eps->get_len())
                    break;
            }
            index_less--;
        }
    
    }
    
    int key_id;
    if (index_less == -1)
        key_id = -1;
    else if (index_less < 2)
        key_id = 0;
    else if (index_less < eps->get_len() - eps->get_gl())
        key_id = min( index_less-2, eps->get_len() - eps->get_gl() - 4 );
    else
        key_id = min( index_less-2, eps->get_len() - 4 );
        
    if (key_id == -1)
    {
        f_extrapolate(e_val, all_outputs);
        return;
    }
        
    while (e_val * Tcm >= sterile_mass/2 && eps->get_value(key_id) * Tcm > sterile_mass/2)
        key_id++;
    while (e_val * Tcm < sterile_mass/2 && eps->get_value(key_id+3) * Tcm > sterile_mass/2)
        key_id--;
        
    f_interpolate(e_val, key_id, all_outputs);
    return;
        
}

void freqs_ntT::f_interpolate(double e_val, int key_id, double* all_outputs)
{
    double log_res, termj;
    for (int i = 0; i < 6; i++)
    {
        for (int j = 0; j < 4; j++)
            if (values[i * num_bins + key_id+j] == 0)
                all_outputs[i] = 0;
        log_res = 0;
        for (int j = 0; j < 4; j++)
        {
            termj = 1;
            for (int k = 0; k < 4; k++)
                if (j != k)
                    termj *= (e_val - eps->get_value(key_id+k))/(eps->get_value(key_id+j) - eps->get_value(key_id+k));
            termj *= log(values[i * num_bins + j]);
            log_res += termj;
        }
        
        all_outputs[i] = exp(log_res);
    }
}

void freqs_ntT::f_extrapolate(double e_val, double* all_outputs)
{
    for (int i = 0; i < 6; i++)
    {
        if ( values[(i+1) * num_bins -1] == 0 || values[(i+1) * num_bins - 2] = 0)
            all_outputs[i] = 0;
            
        double A = values[(i+1) * num_bins - 2];
        double k = - log(values[(i+1) * num_bins - 1] / A) / (eps->get_value(num_bins-1) - eps->get_value(num_bins-2));
        
        all_outputs[i] = A * exp(- k * (e_val - eps->get_value(num_bins-2)));
    }
}