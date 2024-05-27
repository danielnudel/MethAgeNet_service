import streamlit as st
from io import StringIO
import pandas as pd

from pages.predictor.pre_process_single_hist import hist_cohort_pre_processing
from pages.predictor.predictor import predict

st.header('Upload multiple hists here')
st.header('You must upload the .hist files and summary file. The prediction may take a few seconds to start', divider='rainbow')

uploaded_files = st.file_uploader("Choose hist files", accept_multiple_files=True)
summary = st.file_uploader('Upload summary file')

if uploaded_files is not None and summary is not None:
    print('predicting')
    files_d = {}
    for file in uploaded_files:
        files_d[file.name] = StringIO(file.getvalue().decode("utf-8"))

    all_hists_by_sample_d = hist_cohort_pre_processing(files_d, StringIO(summary.getvalue().decode("utf-8")))
    for name in all_hists_by_sample_d:
        ages = [name]
        columns = ['sample_name']
        for marker in all_hists_by_sample_d[name]:
            hist = all_hists_by_sample_d[name][marker]
            age, _, _, _, _ = predict(marker, pd.DataFrame(hist))
            ages.append(age)
            columns.append(marker)
        df = pd.DataFrame([ages], columns=columns)
        st.write(df)
