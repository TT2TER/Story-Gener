import math
import collections
from rouge import Rouge 
from nltk.translate.bleu_score import sentence_bleu
def bleu_nltk(predict,ground_truth):
    bleu_score = sentence_bleu(ground_truth,predict)
    return bleu_score
def bleu(pred_seq, label_seq, k: int = 4):
    """计算BLEU
    输入
    pred_seq = ["I", "like", "cats"]
    label_seq = ["I", "love", "cats"]
    score = bleu(pred_seq, label_seq)
    k为bleu-k
    """

    # 指数
    pred_tokens, label_tokens = list(pred_seq), list(label_seq)  # 深拷贝
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    # 计算BP（Brevity Penalty）
    score = math.exp(min(0, 1 - len_label / len_pred))

    for n in range(1, k + 1):
        num_matches, label_subs = 0, collections.defaultdict(int)
        # 统计标签中n-gram的出现次数
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        # 计算预测中与标签匹配的n-gram数量
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        # 计算BLEU的每个n-gram的贡献，乘到总分上
        score *= math.pow(num_matches / (len_pred - n + 1), 1 / n)

    return score

def rouge(hypothesis, reference):
    '''ouge-n按照论文的说法，是取r（f是取了p和r的调和平均）；rouge-l和其他是取f。不过一般都取f。f是同时考虑了p和r的。'''
    rouger = Rouge()
    # hypothesis = "the #### transcript is a written version of each day 's cnn student news program use this transcript to he    lp students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s you     saw on cnn student news"
    # print(hypothesis)
    # reference = "this page includes the show transcript use the transcript to help students with reading comprehension and     vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teac    her or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news"
    # print(reference)
    scores = rouger.get_scores(hypothesis, reference)
    return scores[0]["rouge-l"]["f"]

if __name__ == "__main__":
    # print(bleu_nltk("I like cats","I love cats"))
    print(rouge("Tommy was very close to his dad and loved him greatly. He was a big fan of his favorite team. Tom was always looking for a new team to play for. His dad was the best player he could find. Tommy was able to find a team that he liked and play with.","Tommy was very close to his dad and loved him greatly. His was a cop and was shot and killed on duty. Tommy cried in his mother's arms at the funeral. Tommy suddenly woke up in a cold sweat. Realizing he had just had a bad dream, he went and hugged his dad."))

