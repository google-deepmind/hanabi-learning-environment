# pylint: disable=missing-module-docstring too-few-public-methods, pointless-string-statement, fixme


class BayesianActionResult:
    '''BayesianAction result'''
    def __init__(self, sampled_action:int, categorical) -> None:
        self.sampled_action = sampled_action
        self.categorical = categorical
