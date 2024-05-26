import streamlit as st
from io import StringIO
import pandas as pd

from pages.predictor.pre_process_single_hist import hist_from_multiple_dfs_pre_processing, concat
from pages.predictor.predictor import predict

st.write("Upload multiple hists here")
uploaded_files = st.file_uploader("Choose a hist file", accept_multiple_files=True)
files_s = []
for file in uploaded_files:
    files_s.append(StringIO(file.getvalue().decode("utf-8")))
if len(files_s) > 0:
    dict_by_loci = hist_from_multiple_dfs_pre_processing(files_s)
    if not dict_by_loci:
        st.write('Something is wrong')
    final_dfs = concat(dict_by_loci)
    if not final_dfs:
        st.write("No good markers")
    for df in dict_by_loci:
        final_dfs[df] = dict_by_loci[df][0]
    for marker in final_dfs:
        hist = final_dfs[marker]
        df = pd.DataFrame([[marker] + list(predict(marker, pd.DataFrame(hist)))], columns=['marker', 'age', 'std', 'p_25', 'p_50', 'p_75'])
        st.table(df)