import streamlit as st
import pandas as pd
from pages.predictor.pre_process_single_hist import hist_from_df_pre_processing
from pages.predictor.predictor import predict

st.header("Enter the reads manually here")
loci_read_lenth = {'ELOVL2_6': 9, 'C1orf132': 8, 'FHL2': 9, 'CCDC102B': 4}
st.write('The loci reads must be of length:')
st.write(str(loci_read_lenth))
loci_u = st.selectbox("Loci", ('ELOVL2_6', 'C1orf132', 'FHL2', 'CCDC102B'))
df = pd.DataFrame([['C' * loci_read_lenth[loci_u], 1]], columns=['read', 'Count'])
edited_df = st.data_editor(df, num_rows="dynamic")

def predict_from_df():
    hist, total_num_reads, loci = hist_from_df_pre_processing(edited_df, loci_u)
    if hist.empty:
        st.write(f'Loci {loci} is not in the list of models')
    df_to_show = pd.DataFrame([predict(loci, hist)], columns=['age', 'std', 'p_25', 'p_50', 'p_75'])
    st.table(df_to_show)

st.button('Predict', on_click=predict_from_df)