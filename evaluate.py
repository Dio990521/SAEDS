from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

file_generate_emo = 'C:/Users/Willy/Desktop/emoDS_emotions.txt'
file_truth_emo = 'C:/Users/Willy/Desktop/ecm_tru_emo.txt'


generate = []
truth = []
with open(file_generate_emo, "r", encoding="utf-8") as f:
    for line in f.readlines():
        generate.append(int(line.strip()))
        
with open(file_truth_emo, "r", encoding="utf-8") as f:
    for line in f.readlines():
        truth.append(int(line.strip()))
        
with open('C:/Users/Willy/Desktop/generate_line.txt', "r", encoding="utf-8") as f:
    for line in f.readlines():
        id.append(int(line.strip()))
real = []
for i in range(len(truth)):
    for j in range(len(id)):
        if i == id[j]:
            real.append(truth[i])
            break
            
count = 0      
for i in range(len(real)):
    if generate[i] == truth[i]:
        count += 1
print('accuracy: ', count/len(generate))


file_generate_sentence = 'C:/Users/Willy/Desktop/test_response.txt'
file_reference_sentence = 'C:/Users/Willy/Desktop/stc_reference_ecm.txt'
generate = []
truth = []
id = []
with open(file_generate_sentence, "r", encoding="utf-8") as f:
    for line in f.readlines():
        generate.append(line.strip())
        
with open(file_reference_sentence, "r", encoding="utf-8") as f:
    for line in f.readlines():
        truth.append(line.strip())
with open('C:/Users/Willy/Desktop/generate_line.txt', "r", encoding="utf-8") as f:
    for line in f.readlines():
        id.append(int(line.strip()))
smooth = SmoothingFunction()
score = 0
real = []
for i in range(len(truth)):
    for j in range(len(id)):
        if i == id[j]:
            real.append(truth[i])
            break
print(len(generate))
for i in range(len(generate)):
    if i > len(real) - 1:
        break
    score += sentence_bleu([truth[i]], generate[i], smoothing_function=smooth.method1)
print('bleu: ', (score/len(generate)) * 100)