from wettbewerb import*
from predict import*
#load files and compute the features

ecg_leads,ecg_labels,fs,ecg_names=load_references(folder='test_examples/')

#Make predictions and save the file

save_predictions(predict_labels(ecg_leads,fs,ecg_names))