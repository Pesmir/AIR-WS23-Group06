from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def graphs(scores, ylabel, title):
    models = ['BERT', 'BERT finetuned', 'XGBoost', 'XGBoost + Word2Vec']
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsalmon']

    plt.figure(figsize=(10, 7))
    bars = plt.bar(models, scores, color=colors)

    for bar, score in zip(bars, scores):
        plt.text(bar.get_x() + bar.get_width() / 2 - 0.1, bar.get_height() + 1, str(score), fontsize=10)

    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(ylabel)
    plt.subplots_adjust(bottom=0.25)

    plt.xticks(rotation=45, ha='right')


    plt.show()


class Evaluation:
    def __init__(self, y_result, y_pred):
        self.y_result = y_result
        self.y_pred = y_pred

    def mae_scores(self):
        scores = []
        for y in self.y_pred:
            scores.append(mean_absolute_error(self.y_result, y))
        graphs(scores, 'MAE', 'MAE Scores of Different Models')
        return scores

    def mse_scores(self):
        scores = []
        for y in self.y_pred:
            scores.append(mean_squared_error(self.y_result, y))
        graphs(scores, 'MSE', 'MSE Scores of Different Models')
        return scores
