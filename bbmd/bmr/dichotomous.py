from .base import DichotomousBase


class Extra(DichotomousBase):

    DUAL_TYPE = 'Extra'

    def get_bmr_vector(self, model):
        return model.extra_risk(self.bmr)


class Added(DichotomousBase):

    DUAL_TYPE = 'Added'

    def get_bmr_vector(self, model):
        return model.added_risk(self.bmr)
