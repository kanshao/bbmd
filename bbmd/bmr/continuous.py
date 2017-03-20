from .base import CentralTendencyBase, HybridBase


class CentralTendencyRelativeChange(CentralTendencyBase):

    ADVERSITY_VERBOSE_NAME = 'relative change'

    def get_bmr_vector(self, model):
        control = model.get_control_vector()
        if (model.response_direction == 1):
            vcutoff = control * (1. + self.adversity_value)
        else:
            vcutoff = control * (1. - self.adversity_value)
        return model.calc_central_tendency(vcutoff)

    def get_adversity_domain(self, domains):
        return domains['relative_change_domain']


class CentralTendencyAbsoluteChange(CentralTendencyBase):

    ADVERSITY_VERBOSE_NAME = 'absolute change'

    def get_bmr_vector(self, model):
        control = model.get_control_vector()
        if (model.response_direction == 1):
            vcutoff = control + self.adversity_value
        else:
            vcutoff = control - self.adversity_value
        return model.calc_central_tendency(vcutoff)

    def get_adversity_domain(self, domains):
        return domains['absolute_change_domain']


class CentralTendencyCutoff(CentralTendencyBase):

    ADVERSITY_VERBOSE_NAME = 'cutoff'

    def get_bmr_vector(self, model):
        return model.calc_central_tendency(self.adversity_value)

    def get_adversity_domain(self, domains):
        return domains['cutoff_domain']


class HybridControlPercentileExtra(HybridBase):

    ADVERSITY_VERBOSE_NAME = 'percentile'
    DUAL_TYPE = 'Extra'

    def get_bmr_vector(self, model):
        return self.calc_bmd_quantile_hybrid(model, isExtra=True)

    def get_adversity_domain(self, domains):
        return domains['quantile_domain']


class HybridControlPercentileAdded(HybridBase):

    ADVERSITY_VERBOSE_NAME = 'percentile'
    DUAL_TYPE = 'Added'

    def get_bmr_vector(self, model):
        return self.calc_bmd_quantile_hybrid(model, isExtra=False)

    def get_adversity_domain(self, domains):
        return domains['quantile_domain']


class HybridAbsoluteCutoffExtra(HybridBase):

    ADVERSITY_VERBOSE_NAME = 'cutoff'
    DUAL_TYPE = 'Extra'

    def get_bmr_vector(self, model):
        return self.calc_bmd_cutoff_hybrid(model, isExtra=True)

    def get_adversity_domain(self, domains):
        return domains['cutoff_domain_hybrid']


class HybridAbsoluteCutoffAdded(HybridBase):

    ADVERSITY_VERBOSE_NAME = 'cutoff'
    DUAL_TYPE = 'Added'

    def get_bmr_vector(self, model):
        return self.calc_bmd_cutoff_hybrid(model, isExtra=False)

    def get_adversity_domain(self, domains):
        return domains['cutoff_domain_hybrid']
