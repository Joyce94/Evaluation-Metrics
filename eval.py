import os
import torch
import numpy as np

class Entity():
    def __init__(self, start, end, category):
        super(Entity, self).__init__()
        self.start = start
        self.end = end
        self.category = category

    def equal(self, entity):
        return self.start == entity.start and self.end == entity.end and self.category == entity.category

    def match(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return len(span.intersection(entity_span)) and self.category == entity.category

    def propor_score(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return float(len(span.intersection(entity_span))) / float(len(span))

    def get_score(self, entity):
        span = set(range(int(self.start), int(self.end) + 1))
        entity_span = set(range(int(entity.start), int(entity.end) + 1))
        return span, entity_span


def Extract_entity(labels, category_set, prefix_array):
    idx = 0
    ent = []
    while (idx < len(labels)):
        if (is_start_label(labels[idx], prefix_array)):
            idy = idx
            endpos = -1
            while (idy < len(labels)):
                if not is_continue(labels[idy], labels[idx], prefix_array, idy - idx):
                    endpos = idy - 1
                    break
                endpos = idy
                idy += 1
            category = cleanLabel(labels[idx], prefix_array)
            # print(category)
            entity = Entity(idx, endpos, category)
            ent.append(entity)
            idx = endpos
        idx += 1
    category_num = len(category_set)
    category_list = [e for e in category_set]
    # print(category_list)
    entity_group = []
    for i in range(category_num):
        entity_group.append([])
    # print(entity_group)
    for id, c in enumerate(category_list):
        for entity in ent:
            if entity.category == c:
                entity_group[id].append(entity)
    return set(ent), entity_group

def is_start_label(label, prefix_array):
    if len(label) < 3:
        return False
    return (label[0] in prefix_array[0]) and (label[1] == '-')

def is_continue(label, startLabel, prefix_array, distance):
    if distance == 0:
        return True
    if len(label) < 3 or label == '<pad>' or label == '<start>':
        return False
    if distance != 0 and is_start_label(label, prefix_array):
        return False
    if (startLabel[0] == 's' or startLabel[0] == 'S') and startLabel[1] == '-':
        return False
    if cleanLabel(label, prefix_array) != cleanLabel(startLabel, prefix_array):
        return False
    return True

def Extract_category(label2id, prefix_array):
    prefix = [e for ele in prefix_array for e in ele]
    category_list = []
    for key in label2id:
        if '-' in key:
            category_list.append(cleanLabel(key, prefix))
    # print(category_list)
    new_list = list(set(category_list))
    new_list.sort(key=category_list.index)
    return new_list

def cleanLabel(label, prefix_array):
    prefix = [e for ele in prefix_array for e in ele]
    if len(label) > 2 and label[1] == '-':
        if label[0] in prefix:
            return label[2:]
    return label

def read_file(read_path):
    file = open(read_path, 'r')
    content = file.readlines()
    labels = []
    for i in range(2, len(content[0].strip().split('['))):
        middle = content[0].strip().split('[')[i].strip()[:-2].split('\'')
        for e in middle:
            if e == '' or e == ', ':
                middle.remove(e)
        labels.append([i for i in middle])
    file.close()
    return labels

def createAlphabet_labeler(label):
    id2label = []
    id2label.append('<start>')
    for index in range(len(label)):
        for w in label[index]:
            if w not in id2label:
                id2label.append(w)
        id2label.append('<pad>')
    id2label = set(id2label)
    id2label = [e for e in id2label]
    return id2label


class Eval():
    def __init__(self, category_set, dataset_num):
        self.category_set = category_set
        self.dataset_sum = dataset_num

    def clear(self):
        self.real_num = 0
        self.predict_num = 0
        self.correct_num = 0
        self.correct_num_p = 0

    def set_eval_var(self):
        self.precision_c = []
        self.recall_c = []
        self.f1_score_c = []

        category_num = len(self.category_set)
        self.B = []
        b = list(range(4))
        for i in range(category_num + 1):
            bb = [0 for e in b]
            self.B.append(bb)

    def Exact_match_my(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        # correct_num = 0
        for p in predict_set:
            for g in gold_set:
                if p.equal(g):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num)
        return result

    def Exact_match(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for i,ent in enumerate(predict_set):
            # gold_count = [0]*self.gold_num
            for id, e in enumerate(gold_set):
                # if gold_count[id] == 0:
                if e.equal(ent):
                    self.correct_num += 1
        result = (self.gold_num, self.predict_num, self.correct_num)
        return result

    def Binary_evaluate_my(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += 1
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += 1
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def Binary_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for i,ent in enumerate(predict_set):
            # gold_count = [0]*self.gold_num
            for id, e in enumerate(gold_set):
                # if gold_count[id] == 0:
                if e.match(ent):
                    self.correct_num += 1
                    self.correct_num_p += 1
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def Propor_evaluate_my(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for p in predict_set:
            for g in gold_set:
                if p.match(g):
                    self.correct_num_p += p.propor_score(g)
                    break
        for g in gold_set:
            for p in predict_set:
                if g.match(p):
                    self.correct_num += g.propor_score(p)
                    break
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def Propor_evaluate(self, predict_set, gold_set):
        self.clear()
        self.gold_num = len(gold_set)
        self.predict_num = len(predict_set)
        for i,ent in enumerate(predict_set):
            # gold_count = [0]*self.gold_num
            for id, e in enumerate(gold_set):
                # if gold_count[id] == 0:
                if e.match(ent):
                    span, entity_span = e.get_score(ent)
                    self.correct_num += float(len(span.intersection(entity_span))) / float(len(span))
                    self.correct_num_p += float(len(span.intersection(entity_span))) / float(len(entity_span))
        result = (self.gold_num, self.predict_num, self.correct_num, self.correct_num_p)
        return result

    def calc_f1_score(self, eval_type):
        category_list = [e for e in self.category_set]
        category_num = len(self.category_set)
        if eval_type == 'Exact':
            for iter in range(category_num):
                precision_ca, recall_ca, f1_score_ca, real_num_ca, predict_num_ca, correct_num_ca = self.get_f1_score_e(self.B[iter + 1][0], self.B[iter + 1][1], self.B[iter + 1][2])
                self.precision_c.append(precision_ca)
                self.recall_c.append(recall_ca)
                self.f1_score_c.append(f1_score_ca)
        else:
            for iter in range(category_num):
                precision_ca, recall_ca, f1_score_ca, real_num_ca, predict_num_ca, correct_num_r, correct_num_p = self.get_f1_score(self.B[iter + 1][0], self.B[iter + 1][1], self.B[iter + 1][2], self.B[iter + 1][3])
                self.precision_c.append(precision_ca)
                self.recall_c.append(recall_ca)
                self.f1_score_c.append(f1_score_ca)

        # print(eval_type + ' NER')
        if os.path.exists(write_path):
            file = open(write_path, "a", encoding='utf-8')
        else:
            file = open(write_path, "w", encoding='utf-8')
        file.write(eval_type + ' NER')
        file.write('\n')
        file.close()
        for index in range(category_num):
            if eval_type == 'Exact':
                if os.path.exists(write_path):
                    file = open(write_path, "a", encoding='utf-8')
                else:
                    file = open(write_path, "w", encoding='utf-8')
                file.write('{}: Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f}'.format(category_list[index], self.B[index + 1][2], self.B[index + 1][0],(self.recall_c[index] * 100), self.B[index + 1][2],self.B[index+1][1], (self.precision_c[index] * 100),(self.f1_score_c[index]*100)))
                file.write('\n')
                file.close()

                # print('{}: Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f}'.format(category_list[index], self.B[index + 1][2], self.B[index + 1][0],(self.recall_c[index] * 100), self.B[index + 1][2],self.B[index+1][1], (self.precision_c[index] * 100),(self.f1_score_c[index]*100)))
            else:
                if os.path.exists(write_path):
                    file = open(write_path, "a", encoding='utf-8')
                else:
                    file = open(write_path, "w", encoding='utf-8')
                file.write('{}: Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f}'.format(
                    category_list[index], self.B[index + 1][2], self.B[index + 1][0], (self.recall_c[index] * 100),
                    self.B[index + 1][3], self.B[index + 1][1], (self.precision_c[index] * 100),
                    (self.f1_score_c[index] * 100)))
                file.write('\n')
                file.close()

                # print('{}: Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f}'.format(category_list[index], self.B[index + 1][2], self.B[index + 1][0], (self.recall_c[index] * 100),self.B[index + 1][3], self.B[index + 1][1], (self.precision_c[index] * 100),(self.f1_score_c[index] * 100)))

        return self.precision_c, self.recall_c, self.f1_score_c

    def calc_f1_score_all(self):
        precision, recall, f1_score, real_num, predict_num, correct_num = self.get_f1_score_e(self.B[0][0], self.B[0][1], self.B[0][2])
        if os.path.exists(write_path):
            file = open(write_path, "a", encoding='utf-8')
        else:
            file = open(write_path, "w", encoding='utf-8')
        file.write('Exact Match:'+'\n')
        file.write('Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f} '.format(correct_num, real_num, (recall * 100),correct_num, predict_num, (precision * 100),(f1_score*100)))
        file.write('\n')
        file.close()

        # print('Exact Match:')
        # print('Recall: P={:.6f}/{:.6f}={:.2f}, Precision: P={:.6f}/{:.6f}={:.2f}, Fmeasure: {:.2f} '.format(correct_num, real_num, (recall * 100),correct_num, predict_num, (precision * 100),(f1_score*100)))
        return precision, recall, f1_score

    def overall_evaluate(self, predict_set, gold_set, eval_type):
        if eval_type == 'Exact':
            return self.Exact_match_my(predict_set, gold_set)
        elif eval_type == 'Binary':
            return self.Binary_evaluate_my(predict_set, gold_set)
        elif eval_type == 'Prop':
            return self.Propor_evaluate_my(predict_set, gold_set)

    def eval(self, gold_labels, predict_labels, eval_type, prefix_array):
        for index in range(len(gold_labels)):
            gold_set, gold_entity_group = Extract_entity(gold_labels[index], self.category_set, prefix_array)
            predict_set, pre_entity_group = Extract_entity(predict_labels[index], self.category_set, prefix_array)
            result = self.overall_evaluate(predict_set, gold_set, eval_type)  # g,p,c
            # print(result)
            for i in range(len(result)):
                self.B[0][i] += result[i]
            for iter in range(len(self.category_set)):
                result = self.overall_evaluate(pre_entity_group[iter], gold_entity_group[iter], eval_type)
                for i in range(len(result)):
                    self.B[iter + 1][i] += result[i]

    def get_f1_score_e(self, real_num, predict_num, correct_num):
        if predict_num != 0:
            precision = correct_num / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0.0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        # result = (precision, recall, f1_score)
        return precision, recall, f1_score, real_num, predict_num, correct_num

    def get_f1_score(self, real_num, predict_num, correct_num_r, correct_num_p):
        if predict_num != 0:
            precision = correct_num_p / predict_num
        else:
            precision = 0.0
        if real_num != 0:
            recall = correct_num_r / real_num
        else:
            recall = 0.0
        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * precision * recall / (precision + recall)
        # result = (precision, recall, f1_score)
        return precision, recall, f1_score, real_num, predict_num, correct_num_r, correct_num_p


def read_sentence_labeler(path):
    words = []
    labels = []
    sen = ' ';lab = ' '
    with open(path, 'r', encoding='utf-8') as fin:
        for id, line in enumerate(fin.readlines()):
            if line == '\n':
                words.append(sen.strip())
                labels.append(lab.strip())
                sen = ' ';lab = ' '
            else:
                sentence = line.strip().split(' ')
                if sentence[0] == 'token':
                    sen = sen + sentence[1] + ' '
                    lab = lab + sentence[-1] + ' '
    return words, labels

def read_corpus_labeler(corpus_data, corpus_labels):
    data = []
    labels = []
    for i in range(len(corpus_data)):
        text = corpus_data[i].strip()
        label = corpus_labels[i].strip()
        labels.append(label.split())
        data.append(text.split())
    return data, labels


if __name__ == "__main__":
    prefix_array = [['b', 'B', 's', 'S'], ['m', 'M', 'e', 'E']]
    eval_type = ['Exact', 'Binary', 'Prop']
    # category_list = ['DSE', 'AGENT', 'TARGET']
    # 动态获取类别数目
    gold_path = 'results-yle/my_data_0/dev_0.txt'
    gold_words, gold_labels = read_sentence_labeler(gold_path)
    gold_data, gold_labels = read_corpus_labeler(gold_words, gold_labels)
    label_list = createAlphabet_labeler(gold_labels)
    category_list = Extract_category(label_list, prefix_array)
    # print(category_list)

    perm_list = list(range(10))
    write_path = './output.txt'
    # print(write_path)
    if os.path.exists(write_path):
        os.remove(write_path)
    recode = []
    for id in range(len(eval_type)):
        eval_l = []
        for j in range(len(category_list)):
            category_l = []
            for k in range(3):
                category_l.append([])
            eval_l.append(category_l)
        recode.append(eval_l)
    recode_overall = [[],[],[]]

    for id in perm_list:
        gold_path = 'results-yle/my_data_'+str(id)+'/dev_'+str(id)+'.txt'
        # print(gold_path)
        predict_path = 'results-yle/my_data_'+str(id)+'/dev_'+str(id)+'.txt.b5.miwa.out'
        # print(predict_path)
        gold_words, gold_labels = read_sentence_labeler(gold_path)
        predict_words, predict_labels = read_sentence_labeler(predict_path)
        count = 0
        gold_words_new = []
        gold_labels_new = []
        for id, e in enumerate(gold_words):
            if e in predict_words:
                count += 1
                gold_words_new.append(e)
                gold_labels_new.append(gold_labels[id])

        gold_data, gold_labels = read_corpus_labeler(gold_words_new, gold_labels_new)
        predict_data, predict_labels = read_corpus_labeler(predict_words, predict_labels)
        # label_list = createAlphabet_labeler(gold_labels)
        # category_list = Extract_category(label_list, prefix_array)
        dataset_num = len(gold_labels)
        if os.path.exists(write_path):
            file = open(write_path, "a", encoding='utf-8')
        else:
            file = open(write_path, "w", encoding='utf-8')
        s = 'Reach: '+str(dataset_num)+', Total: '+str(dataset_num)
        # print(s)
        file.write(s)
        file.write('\n')
        s2 = 'Evaluating: '+predict_path
        file.write(s2)
        file.write('\n')
        file.close()
        evaluation = Eval(category_list, dataset_num)
        for i in range(len(eval_type)):
            evaluation.set_eval_var()
            evaluation.eval(gold_labels, predict_labels, eval_type[i], prefix_array)
            precision_c, recall_c, f1_score_c = evaluation.calc_f1_score(eval_type[i])
            for j in range(len(category_list)):
                recode[i][0][j].append(recall_c[j])
            for j in range(len(category_list)):
                recode[i][1][j].append(precision_c[j])
            for j in range(len(category_list)):
                recode[i][2][j].append(f1_score_c[j])

        evaluation.set_eval_var()
        evaluation.eval(gold_labels, predict_labels, 'Exact', prefix_array)
        precision, recall, f1_score = evaluation.calc_f1_score_all()
        recode_overall[0].append(recall)
        recode_overall[1].append(precision)
        recode_overall[2].append(f1_score)
        # print('\n')
        if os.path.exists(write_path):
            file = open(write_path, "a", encoding='utf-8')
        else:
            file = open(write_path, "w", encoding='utf-8')
        file.write('\n')
        file.close()

    # print('Summary')
    if os.path.exists(write_path):
        file = open(write_path, "a", encoding='utf-8')
    else:
        file = open(write_path, "w", encoding='utf-8')
    file.write('Summary')
    file.write('\n')
    file.close()

    for i in range(len(eval_type)):
        # print(eval_type[i]+' NER')
        if os.path.exists(write_path):
            file = open(write_path, "a", encoding='utf-8')
        else:
            file = open(write_path, "w", encoding='utf-8')
        file.write(eval_type[i]+' NER')
        file.write('\n')
        file.close()
        summary_list = []
        for w in range(len(category_list)):
            summary_list.append([])
            for q in range(3):
                summary_list[w].append([])
        for j in range(3):
            for k in range(len(category_list)):
                summary_list[k][j].append(np.mean(recode[i][j][k]))
                summary_list[k][j].append(np.std(recode[i][j][k]))
        # print(summary_list)
        for index in range(len(category_list)):
            # print('{}: Recall(mean = {:.6f}, deri = {:.6f}), Precision(mean = {:.6f}, deri = {:.6f}), Fscore(mean = {:.6f}, deri = {:.6f})'.format(category_set[index], summary_list[index][0][0], summary_list[index][0][1], summary_list[index][1][0],summary_list[index][1][1],summary_list[index][2][0],summary_list[index][2][1]))
            if os.path.exists(write_path):
                file = open(write_path, "a", encoding='utf-8')
            else:
                file = open(write_path, "w", encoding='utf-8')
            file.write('{}: Recall(mean = {:.6f}, deri = {:.6f}), Precision(mean = {:.6f}, deri = {:.6f}), Fscore(mean = {:.6f}, deri = {:.6f})'.format(category_list[index], summary_list[index][0][0], summary_list[index][0][1], summary_list[index][1][0],summary_list[index][1][1], summary_list[index][2][0],summary_list[index][2][1]))
            file.write('\n')
            file.close()

    overall = [[],[],[]]
    # print('Exact NER, overall')
    if os.path.exists(write_path):
        file = open(write_path, "a", encoding='utf-8')
    else:
        file = open(write_path, "w", encoding='utf-8')
    file.write('Exact NER, overall')
    file.write('\n')
    file.close()
    for id in range(3):
        overall[id].append(np.mean(recode_overall[id]))
        overall[id].append(np.std(recode_overall[id]))
    # print('ner: Recall(mean = {:.6f}, deri = {:.6f}), Precision(mean = {:.6f}, deri = {:.6f}), Fscore(mean = {:.6f}, deri = {:.6f})'.format(overall[0][0], overall[0][1], overall[1][0], overall[1][1], overall[2][0], overall[2][1]))
    if os.path.exists(write_path):
        file = open(write_path, "a", encoding='utf-8')
    else:
        file = open(write_path, "w", encoding='utf-8')
    file.write('ner: Recall(mean = {:.6f}, deri = {:.6f}), Precision(mean = {:.6f}, deri = {:.6f}), Fscore(mean = {:.6f}, deri = {:.6f})'.format(overall[0][0], overall[0][1], overall[1][0], overall[1][1], overall[2][0], overall[2][1]))
    file.close()
