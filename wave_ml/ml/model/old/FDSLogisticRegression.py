from multipledispatch import dispatch
from sklearn.linear_model import LogisticRegression

from wave_ml.ml.model.old.FDSModel import FDSModel
from wave_ml.ml.common.CommonUtil import CommonUtil as util
from wave_ml.ml.common.CommonProperties import CommonProperties as prop
from wave_ml.fds.common.LoggingHandler import LoggingHandler

logger = LoggingHandler(f"{prop().get_result_log_file_path()}{util().now_type3()}_FDSLogisticRegression", "a", "DEBUG")


class FDSLogisticRegression(FDSModel):

    """
    penalty     - None: no penalty is added;
                - 'l2': add a L2 penalty term and it is the default choice;
                - 'l1': add a L1 penalty term;
                - 'elasticnet': both L1 and L2 penalty terms are added.
    dual : bool, default=False
                - Dual or primal formulation. Dual formulation is only implemented for l2 penalty with liblinear solver.
                Prefer dual=False when n_samples > n_features.

    tol : float, default=1e-4
                - Tolerance for stopping criteria.

    C : float, default=1.0
                - Inverse of regularization strength; must be a positive float.
                Like in support vector machines, smaller values specify stronger regularization.

    fit_intercept : bool, default=True
                - Specifies if a constant (a.k.a. bias or intercept) should be added to the decision function.

    intercept_scaling : float, default=1
                - Useful only when the solver ‘liblinear’ is used and self.fit_intercept is set to True.
                In this case, x becomes [x, self.intercept_scaling], i.e. a “synthetic” feature with constant value equal to intercept_scaling is appended to the instance vector.
                The intercept becomes intercept_scaling * synthetic_feature_weight.
                - Note! the synthetic feature weight is subject to l1/l2 regularization as all other features.
                To lessen the effect of regularization on synthetic feature weight (and therefore on the intercept) intercept_scaling has to be increased.

    class_weight : dict or ‘balanced’, default=None
                - Weights associated with classes in the form {class_label: weight}.
                If not given, all classes are supposed to have weight one.
                - The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
                - Note that these weights will be multiplied with sample_weight (passed through the fit method) if sample_weight is specified.
                - New in version 0.17: class_weight=’balanced’

    random_state : int, RandomState instance, default=None
                - Used when solver == ‘sag’, ‘saga’ or ‘liblinear’ to shuffle the data. See Glossary for details.

    solver : {‘lbfgs’, ‘liblinear’, ‘newton-cg’, ‘newton-cholesky’, ‘sag’, ‘saga’}, default=’lbfgs’
                - Algorithm to use in the optimization problem. Default is ‘lbfgs’.
                To choose a solver, you might want to consider the following aspects:

                -> For small datasets, ‘liblinear’ is a good choice, whereas ‘sag’ and ‘saga’ are faster for large ones;
                -> For multiclass problems, only ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ handle multinomial loss;
                -> ‘liblinear’ and is limited to one-versus-rest schemes.
                -> ‘newton-cholesky’ is a good choice for n_samples >> n_features, especially with one-hot encoded categorical features with rare categories.
                Note that it is limited to binary classification and the one-versus-rest reduction for multiclass classification.
                Be aware that the memory usage of this solver has a quadratic dependency on n_features because it explicitly computes the Hessian matrix.
    """

    def __init__(self, data):
        self.model = LogisticRegression(random_state=42)
        self.log = logger.get_log()

        super().__init__(self.log, self.model, data)

    @dispatch(int)
    def set_model(self, max_iter):
        self.model = LogisticRegression(max_iter=max_iter, random_state=42)

    @dispatch(float)
    def set_model(self, elasticnet):
        self.model = LogisticRegression(solver='saga', penalty='elasticnet', l1_ratio=elasticnet, random_state=42)

    @dispatch(int, float)
    def set_model(self, max_iter, elasticnet):
        self.model = LogisticRegression(max_iter=max_iter, solver='saga', penalty='elasticnet', l1_ratio=elasticnet, random_state=42)
