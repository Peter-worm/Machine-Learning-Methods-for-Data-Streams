from river import tree, forest, naive_bayes, linear_model, neighbors

class TreeModelFactory:
    def create_model(self):
        return tree.HoeffdingTreeClassifier()
class ARFModelFactory:
    def create_model(self):
        return forest.ARFClassifier(seed=1)
class NaiveBayesFactory:
    def create_model(self):
        return naive_bayes.GaussianNB()
class LogisticRegressionFactory:
    def create_model(self):
        return linear_model.LogisticRegression()
class KNNFactory:
    def create_model(self):
        return neighbors.KNNClassifier()
    
class ModelFactoryProducer:
    factories = {
        "hoeffding_tree": TreeModelFactory,
        "adaptive_rf": ARFModelFactory,
        "naive_bayes": NaiveBayesFactory,
        "logistic_regression": LogisticRegressionFactory,
        "knn": KNNFactory,
    }

    @staticmethod
    def get_factory(model_name: str):
        factory_class = ModelFactoryProducer.factories.get(model_name.lower())
        if factory_class is None:
            raise ValueError(f"No factory found for model: {model_name}")
        return factory_class()