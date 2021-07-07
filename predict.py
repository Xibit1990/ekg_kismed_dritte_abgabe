import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from Preprocess import *
from ProposedCNN import *
def predict_labels(ecg_leads,fs,ecg_names,use_pretrained=False):
    '''
    Parameters
    ----------
    model_name : str
        Dateiname des Models. In Code-Pfad
    ecg_leads : list of numpy-Arrays
        EKG-Signale.
    fs : float
        Sampling-Frequenz der Signale.
    ecg_names : list of str
        eindeutige Bezeichnung für jedes EKG-Signal.
    Returns
    -------
    predictions : list of tuples
        ecg_name und eure Diagnose
    '''

#------------------------------------------------------------------------------
    codes = {0: 'A', 1: 'N', 2: 'O', 3: '~'}
    #denoise signals
    pp = PreprocessingData(fs)
    test_leads = pp.denoise_data(ecg_leads)
    try:
        print('[INFO]*****Creating features')
        features = np.array([pp._logMelFilterbank(sample) for sample in test_leads])
    except:
        print('[ERROR]*****Error creating features')
    print('[INFO]*****Features created')

    print('[INFO]*****Loading model')
    try:
        model = tf.keras.models.load_model('./model/crnn/')
    except:
        print('[ERROR]*****Error loading Model')
    print('[INFO]*****Model loaded')

    print('[INFO]*****Peforming predictions')
    # evaluate the model
    predicted_labels = []
    predicted_labels_raw = model.predict(features)

    [predicted_labels.append(np.argmax(elements)) for elements in predicted_labels_raw.tolist()]
    decoded_labels = (pd.Series(predicted_labels)).map(codes).tolist()

    predictions = list(zip(ecg_names, decoded_labels))
    return predictions