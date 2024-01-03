from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


def graphs(scores, ylabel, title):
    models = ['BERT', 'WordVec', 'XGBoost']
    plt.figure(figsize=(8, 6))
    plt.bar(models, scores, color=['blue', 'green', 'orange'])

    plt.title(title)
    plt.xlabel('Models')
    plt.ylabel(ylabel)

    plt.xticks(rotation=45)
    plt.tight_layout()
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
