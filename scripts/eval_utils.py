import numpy as np 

def eval_absa(y_pred, y_true):
    tp = .0
    fp = .0
    fn = .0
    n_total = 0
    for pred, gold in zip(y_pred, y_true):
        pred = list(set(pred.split(' và ')))
        pred = [aspect.strip() for aspect in pred]
        gold = list(set(gold.split(' và ')))
        gold = [aspect.strip() for aspect in gold]
        n_total += len(gold)
        for aspect in gold:
            if aspect in pred:
                tp += 1
            else:
                fn += 1
        for aspect in pred:
            if aspect not in gold:
                fp += 1

    precision = 0 if tp + fp == 0 else 1.*tp / (tp + fp)
    recall = 0 if tp + fn == 0 else 1.*tp / (tp + fn)
    f1 = 0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
    acc = tp / n_total
    print(f"tp: {tp}, fp: {fp}, fn: {fn}")
    print(f"p: {precision}, r: {recall}, f1: {f1}, acc: {acc}")
    return {'precision': precision, 'recall': recall, 'f1': f1, "acc": acc}