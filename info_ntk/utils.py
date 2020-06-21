import jax.numpy as np
import pandas as pd
def create_df(result, ts, sigs):
    sigs_ts =np.array([[(sig, t) for sig in sigs] for t in ts])
    sigs_flattend  = sigs_ts[:, :, 0].flatten()
    ts_flattend  = sigs_ts[:, :, 1].flatten()
    df = pd.DataFrame(columns=['name', 't', 'w_std', 'value'])
    for name, values in result.items():
        flattend_values = values.flatten()
        df_rows =  pd.DataFrame(flattend_values, columns = ['value'])
        df_rows['t'] = ts_flattend
        df_rows['name'] = name
        df_rows['w_std'] = sigs_flattend
        df = df.append(df_rows, ignore_index = True)
    return df

def reorder_dict(df):
    dict_new = {}
    for key in df[0]:
        current_values = []
        for i in range(len(df)):
            current_values.append(df[i][key])
        dict_new[key] =  np.array(current_values)
    return dict_new