import numpy as np
from data.cbet_data_loader import cbet_data
from data.semeval2018t3ec.data_loader_semeval2018 import load_sem18_all
from data.toxic_comment.data_loader_toxic_comment import load_tc_all
from data.goemotions.goemotion_loader import goemotion_data
from tqdm import tqdm
import pandas as pd
from utils.tweet_processor import TextProcessor


MAX_LEN_DATA = 50

def load_BMET_data(for_seq2emo=True, load_split=False):
    EMOS = ['anger', 'fear', 'joy', 'sadness', 'surprise', 'thankfulness']
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    # data_pata = 'data/EmoSet_RemoveDup_GloveProcess_OneEmo.csv'
    data_pata = 'data/BMETv0.3.csv'
    df_data = pd.read_csv(data_pata)

    # extract the subset which only contains the full sentences.
    source = []
    target = []
    for index, row in df_data.iterrows():
        next_token = str(row['text']).strip().split()
        if len(next_token) > MAX_LEN_DATA:
            next_token = next_token[:MAX_LEN_DATA]
        source.append(' '.join(next_token))
        if for_seq2emo:
            a_target = [0, 2, 4, 6, 8, 10]
            label = row['label'].split()
            for emo in label:
                a_target[EMOS_DIC[emo]] = EMOS_DIC[emo] * 2 + 1
        else:
            a_target = [0] * len(EMOS)
            label = row['label'].split()
            for emo in label:
                a_target[EMOS_DIC[emo]] = 1
        target.append(a_target)
    if not load_split:
        return source, target, EMOS, EMOS_DIC, 'BMETv0.3'
    else:
        from sklearn.model_selection import ShuffleSplit
        X, y = source, target
        ss = ShuffleSplit(n_splits=1, test_size=0.2, random_state=999)
        ss.get_n_splits(X, y)
        train_index, test_index = next(ss.split(y))
        X_train_dev, X_test = [X[i] for i in train_index], [X[i] for i in test_index]
        y_train_dev, y_test = [y[i] for i in train_index], [y[i] for i in test_index]
    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'BMETv0.3'


def load_sem18_data(for_seq2emo=True, load_split=False):
    text_processor = TextProcessor()
    def __process_data(data, label, EMOS):
        EMOS_DIC = {}
        for idx, emo in enumerate(EMOS):
            EMOS_DIC[emo] = idx
        target = []
        for l in label:
            if for_seq2emo:
                a_target = list(range(len(EMOS)))
                a_target = [x * 2 for x in a_target]
                pos_position = np.where(np.asarray(l) == 1)[0].tolist()
                for pos in pos_position:
                    a_target[pos] = pos * 2 + 1
            else:
                a_target = [0] * len(EMOS)
                pos_position = np.where(np.asarray(l) == 1)[0].tolist()
                for pos in pos_position:
                    a_target[pos] = 1
            target.append(a_target)

        source = []
        for text in tqdm(data):
            text = text_processor.processing_pipeline(text)
            source.append(text)
        return source, target, EMOS, EMOS_DIC

    if not load_split:
        data, label, EMOS = load_sem18_all()
        X, y, EMOS, EMOS_DIC = __process_data(data, label, EMOS)
        return X, y, EMOS, EMOS_DIC, 'sem18'
    else:
        data, label, EMOS = load_sem18_all(is_test=False, load_all=False)
        X_train_dev, y_train_dev, _, _ = __process_data(data, label, EMOS)
        data, label, EMOS = load_sem18_all(is_test=True, load_all=False)
        X_test, y_test, EMOS, EMOS_DIC = __process_data(data, label, EMOS)
        return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'sem18_split'


def load_tc_data(for_seq2emo=True, load_split=False):
    text_processor = TextProcessor()
    def __process_data(data, label, EMOS):
        EMOS_DIC = {}
        for idx, emo in enumerate(EMOS):
            EMOS_DIC[emo] = idx
        target = []
        for l in label:
            if for_seq2emo:
                a_target = list(range(len(EMOS)))
                a_target = [x * 2 for x in a_target]
                pos_position = np.where(np.asarray(l) == 1)[0].tolist()
                for pos in pos_position:
                    a_target[pos] = pos * 2 + 1
            else:
                a_target = [0] * len(EMOS)
                pos_position = np.where(np.asarray(l) == 1)[0].tolist()
                for pos in pos_position:
                    a_target[pos] = 1
            target.append(a_target)

        source = []
        for text in tqdm(data):
            text = text_processor.processing_pipeline(text)
            source.append(text)
        return source, target, EMOS, EMOS_DIC

    if not load_split:
        data, label, EMOS = load_tc_all()
        X, y, EMOS, EMOS_DIC = __process_data(data, label, EMOS)
        return X, y, EMOS, EMOS_DIC, 'tc'
    else:
        data, label, EMOS = load_tc_all(is_test=False, load_all=False)
        X_train_dev, y_train_dev, _, _ = __process_data(data, label, EMOS)
        data, label, EMOS = load_tc_all(is_test=True, load_all=False)
        X_test, y_test, EMOS, EMOS_DIC = __process_data(data, label, EMOS)
        return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, 'tc_split'


def load_cbet_data(for_seq2emo=True):
    text_processor = TextProcessor()
    data, label, EMOS, _ = cbet_data(remove_stop_words=False, preprocess=False, multi=True, vector=True)
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx

    target = []
    for l in label:
        if for_seq2emo:
            a_target = list(range(len(EMOS)))
            a_target = [x * 2 for x in a_target]
            pos_position = np.where(np.asarray(l) == 1)[0].tolist()
            for pos in pos_position:
                a_target[pos] = pos * 2 + 1
        else:
            a_target = [0] * len(EMOS)
            pos_position = np.where(np.asarray(l) == 1)[0].tolist()
            for pos in pos_position:
                a_target[pos] = 1
        target.append(a_target)

    source = []
    print('processing data ...')
    for text in tqdm(data):
        text = text_processor.processing_pipeline(text)
        source.append(text)
    return source, target, EMOS, EMOS_DIC, 'cbet'


def load_goemotions_data():
    X_train_raw, y_train, X_dev_raw, y_dev, X_test_raw, y_test, emo_list = goemotion_data(file_path='data/goemotions')
    X_train_dev_raw = X_train_raw + X_dev_raw
    y_train_dev = y_train + y_dev
    # preprocess
    text_processor = TextProcessor()

    X_train_dev = []
    for text in tqdm(X_train_dev_raw):
        text = text_processor.processing_pipeline(text)
        X_train_dev.append(text)

    X_test = []
    for text in tqdm(X_test_raw):
        text = text_processor.processing_pipeline(text)
        X_test.append(text)

    EMOS = emo_list
    EMOS_DIC = {}
    for idx, emo in enumerate(EMOS):
        EMOS_DIC[emo] = idx
    data_set_name = ''
    return X_train_dev, y_train_dev, X_test, y_test, EMOS, EMOS_DIC, data_set_name

