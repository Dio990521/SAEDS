# only look at BLEU and SkipThought metrics.

from nlgeval import compute_metrics
file_generate = "/local/ssd_1/chengzhang/SA_dialog/dialogue/result/nlpcc2017_tune_output/bs_emo8.txt"
file_truth = '/local/ssd_1/chengzhang/SA_dialog/big_data/nlpcc2017_reference.txt'
generate = []
truth = []
with open(file_generate, "r", encoding="utf-8") as f:
    for line in f.readlines():
        generate.append(line.strip())
        
with open(file_truth, "r", encoding="utf-8") as f:
    for line in f.readlines():
        truth.append([line.strip()])

print(len(truth))
print(len(generate))
print("BLUE score:", compute_metrics(hypothesis=file_generate,
                               references=[file_truth]))