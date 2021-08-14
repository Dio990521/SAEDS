from torchtext.data.metrics import bleu_score
import nltk
from nltk.translate.bleu_score import sentence_bleu
from nlgeval import NLGEval
nlgeval = NLGEval(no_skipthoughts=True, no_glove=True)  # loads the models


def bleu(gold, pred):
    # t = sentence_bleu(gold, pred)
    # print('BLEU:', bleu_score(pred, [[item.split()] for item in gold[0]]))
    metrics_dict = nlgeval.compute_metrics(gold, [' '.join(item) for item in pred])
    # result_str = f"Bleu1 {metrics_dict['Bleu_1']} Bleu2 {metrics_dict['Bleu_2']}"
    return metrics_dict


# >>> from torchtext.data.metrics import bleu_score
# >>> candidate_corpus = [['My', 'full', 'pytorch', 'test'], ['Another', 'Sentence']]
# #>>> references_corpus = [[['My', 'full', 'pytorch', 'test'], ['Completely', 'Different']], [['No', 'Match']]]
# >>> bleu_score(candidate_corpus, references_corpus)

def evaluation(gold, pred):
    pass
