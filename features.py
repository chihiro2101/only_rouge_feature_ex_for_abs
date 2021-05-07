import preprocess
import math
import nltk
import numpy as np
from rouge import Rouge

class METRIC(object):
    def __init__(self, title, raw_sentences, agent, rougeforsentences, abstract,  order_params):
        self.title = title
        self.raw_sentences = raw_sentences
        self.n = len(raw_sentences)
        self.values = agent
        self.rougeforsentences = rougeforsentences
        self.abstract = abstract
        self.order_params = order_params

    # number of sentences in summary
    def O(self):
        return np.sum(self.values)

    def rouge_scores(self):
        rouge1f = 0
        for i in range(self.n):
            if self.values[i] == 1:
                rouge1f += self.rougeforsentences[i]
        return rouge1f/self.O()


    def calculate_rouge_for_summary(self):
        summary = ''
        for i in range(self.n):
            if self.values[i] == 1:
                summary += self.raw_sentences[i] + ' '
        try:
            rouge = Rouge()
            scores = rouge.get_scores(summary, abstract, avg=True)
            rouge1f = scores["rouge-1"]["f"]
            rouge2f = scores["rouge-2"]["f"]
        except Exception:
            rouge1f = 0 
            rouge2f = 0
        return rouge1f, rouge2f



    def fitness(self):
        if self.order_params == 0:
            fit = self.rouge_scores()*100
        elif self.order_params == 1:
            r1, r2 = self.calculate_rouge_for_summary() 
            fit = r1*100
        elif self.order_params == 2:
            rouge1f = 50
            rouge2f = 50
            r1, r2 = self.calculate_rouge_for_summary()
            fit = r1*rouge1f + r2*rouge2f

        return fit


def compute_fitness(title, sentences, agent, rougeforsentences, abstract, order_params):
    metric = METRIC(title, sentences, agent, rougeforsentences, abstract, order_params)
    return metric.fitness()
