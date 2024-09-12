#include "freqs_ntT.hh"

#include <cmath>
#include <iostream>

using std::abs;
using std::cout;
using std::endl;

using std::min;

void freqs_ntT::interp_extrap(double e_val, double Tcm, double* all_outputs)
{
    int index_less = -1;
    
    for(int i = 0; i < eps->get_gel()+1; i++)
    {
        if (e_val == eps->get_value(i))
        {
            for(int j = 0; j < 6; j++)
                all_outputs[j] = values[j * num_bins + i];
            return;
        }
    }
    
    if (e_val < eps->get_min_linspace())
    {
        index_less = int(eps->get_gel() / 2);
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
    
/*        for de in delta_fn:
        for j in range(3):
            if eps_old[key_id+j] <= de/Tcm <= eps_old[key_id+j+1]:
                print(eps_old[key_id], e_val, eps_old[key_id+3])
                return np.sqrt(f_old[key_id+j] * f_old[key_id+j+1])
*/
 /*   for(int i = 0; i < 2; i++)
        for(int j = 0; j < 3; j++)
            if(eps->get_value(key_id+j) <= delta_decays[i] / Tcm && delta_decays[i] / Tcm <= eps->get_value(key_id+j+1))
            {
                for(int k = 0; k < 6; k++)
                    all_outputs[k] = sqrt(values[k * num_bins + key_id + j]) * sqrt(values[k * num_bins + key_id + j + 1]);
                return;
            }*/
    f_interpolate(e_val, key_id, index_less, all_outputs);
    return;
        
}

void freqs_ntT::f_interpolate(double e_val, int key_id, int index_less, double* all_outputs)
{

    double log_res, termj;
    bool zero;
    for (int i = 0; i < 6; i++)
    {
        log_res = 0;
        zero = false;
        for (int j = 0; j < 4; j++)
        {
            if (values[i * num_bins + key_id+j] == 0)
            {
                all_outputs[i] = 0;
                zero = true;
                break;
            }
            termj = 1;
            for (int k = 0; k < 4; k++)
                if (j != k)
                    termj *= (e_val - eps->get_value(key_id+k))/(eps->get_value(key_id+j) - eps->get_value(key_id+k));
            termj *= log(values[i * num_bins + key_id + j]);
            log_res += termj;
        }
        
        if(zero)
            all_outputs[i] = 0;
        else
        {
            all_outputs[i] = exp(log_res);
        /*    if ((all_outputs[i] - values[i * num_bins + index_less]) * (all_outputs[i] - values[i * num_bins + index_less + 1]) > 0)
            {
                cout << e_val << "***" << all_outputs[i] << ", " << values[i*num_bins+index_less] << ", " << values[i*num_bins + index_less+1] << endl;
                all_outputs[i] = sqrt(values[i * num_bins + index_less]) * sqrt(values[i * num_bins + index_less + 1]);
            }*/
        }
//        if(std::isnan(all_outputs[i]))
//            cout << e_val << ", " << key_id << ", " << log_res << endl;
    }
}

void freqs_ntT::f_extrapolate(double e_val, double* all_outputs)
{
    for (int i = 0; i < 6; i++)
    {
        if ( values[(i+1) * num_bins -1] == 0 || values[(i+1) * num_bins - 2] == 0)
            all_outputs[i] = 0;
        else
        {
            double A = values[(i+1) * num_bins - 2];
            double k = - log(values[(i+1) * num_bins - 1] / A) / (eps->get_value(num_bins-1) - eps->get_value(num_bins-2));
                    
            all_outputs[i] = A * exp(- k * (e_val - eps->get_value(num_bins-2)));
        }
    }
}