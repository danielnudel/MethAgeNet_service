import streamlit as st
from io import StringIO
import pandas as pd

from pages.predictor.pre_process_single_hist import hist_pre_processing
from pages.predictor.predictor import predict

st.header("Upload single hist here", divider='rainbow')
uploaded_file = st.file_uploader("Choose a hist file")
if uploaded_file is not None:
    io_file = StringIO(uploaded_file.getvalue().decode("utf-8"))

    hist, total_num_reads, loci = hist_pre_processing(io_file)
    if hist.empty:
        st.write(f'Loci {loci} is not in the list of models')
    else:
        st.write(list(predict(loci, hist)))
        df = pd.DataFrame([loci] + list(predict(loci, hist)), columns=['loci', 'age', 'std', 'p_25', 'p_50', 'p_75'])
        st.table(df)