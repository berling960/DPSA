import importlib

attack_zoo = {
    # Gradient baselines
    'fgsm': ('.gradient.fgsm', 'FGSM'),
    'mifgsm': ('.gradient.mifgsm', 'MIFGSM'),
    'nifgsm': ('.gradient.nifgsm', 'NIFGSM'),
    'aifgtm': ('.gradient.aifgtm', 'AIFGTM'),
    'gifgsm': ('.gradient.gifgsm', 'GIFGSM'),
    'emifgsm': ('.gradient.emifgsm', 'EMIFGSM'),
    'vnifgsm': ('.gradient.vnifgsm', 'VNIFGSM'),
    'vmifgsm': ('.gradient.vmifgsm', 'VMIFGSM'),
    'gra': ('.gradient.gra', 'GRA'),

    # Input-transformation baselines
    'maskblock': ('.input_transformation.maskblock', 'MaskBlock'),
    'usmm': ('.input_transformation.usmm', 'USMM'),
    'admix': ('.input_transformation.admix', 'Admix'),
    'sim': ('.input_transformation.sim', 'SIM'),
    'dim': ('.input_transformation.dim', 'DIM'),
    'ssm_h': ('.input_transformation.ssm_with_tricks', 'SSM_H'),
    'mfi': ('.input_transformation.mfi', 'MFI'),
    'dem': ('.input_transformation.dem', 'DEM'),
    'sia': ('.input_transformation.sia', 'SIA'),
    'bsr': ('.input_transformation.bsr', 'BSR'),
    'ops': ('.input_transformation.ops', 'OPS'),

    # Proposed method
    'dpsa': ('.dpsa', 'DPSA'),
    'lpsa': ('.dpsa', 'DPSA'),  # legacy alias
}


def load_attack_class(attack_name):
    if attack_name not in attack_zoo:
        raise Exception('Unspported attack algorithm {}'.format(attack_name))
    module_path, class_name = attack_zoo[attack_name]
    module = importlib.import_module(module_path, __package__)
    attack_class = getattr(module, class_name)
    return attack_class

def load_attack(attack_name, model_name, model, targeted, **kwargs):
    attack_class = load_attack_class(attack_name)
    # DPSA owns surrogate construction because it supports custom wrappers.
    if attack_name in ('dpsa', 'lpsa'):
        return attack_class(model_name=model_name, targeted=targeted, **kwargs)

    from . import attack as attack_base
    attack_base.set_preloaded_model(model.eval().cuda())
    try:
        attack = attack_class(model_name=model_name, targeted=targeted, **kwargs)
    finally:
        attack_base.set_preloaded_model(None)

    if model is not None:
        attack.model = model.eval().cuda()
        attack.device = next(attack.model.parameters()).device
    return attack


__version__ = '1.0.0'
