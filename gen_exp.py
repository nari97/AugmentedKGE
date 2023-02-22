def run():
    filename = 'Exp1'

    with open(filename, 'w') as f:
        for model_name in ['analogy', 'aprile', 'boxe', 'combine', 'complex', 'crosse_interactions',
                           'crosses_nointeractions', 'cycle', 'dense', 'distmult_notanh', 'distmult_tanh', 'hake_both',
                           'hake_modonly', 'harotate', 'hole', 'kg2e_kldivergence', 'kg2e_elikelihood', 'lineare',
                           'makr', 'manifolde_hyperplane', 'manifolde_sphere', 'mde', 'mrotate', 'nage_su2', 'nage_so3',
                           'pairre', 'protate', 'quatde', 'quate', 'rate', 'rotate', 'rotate3d', 'rotpro', 'se',
                           'simple_both', 'simple_ignr', 'stranse', 'structure', 'toruse_L1', 'toruse_L2', 'toruse_eL2',
                           'transa', 'transat', 'transd', 'transdr', 'transe', 'transh', 'transm', 'transms', 'transr',
                           'transsparse_share', 'transsparse_separate', 'tucker']:
            for dataset in range(0, 8):
                f.write(model_name)
                f.write(',')
                f.write(str(dataset))
                f.write(', ')
                f.write(',0')
                f.write('\n')


if __name__ == '__main__':
    run()