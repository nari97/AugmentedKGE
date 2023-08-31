def run():
    filename = 'RCLarge'

    with open(filename, 'w') as f:
        for model_name in ['analogy', 'aprile', 'atth_atth', 'atth_roth', 'atth_refh', 'atth_atte', 'atth_rote',
                           'atth_refe', 'boxe', 'combine', 'complex', 'cp', 'crosse_interactions',
                           'crosse_nointeractions', 'cycle', 'dense', 'dihedral_4', 'dihedral_6', 'dihedral_8',
                           'distmult_notanh', 'distmult_tanh', 'duale_cross', 'duale_full', 'fivestare', 'geome_2d',
                           'geome_3d', 'geome_plus', 'gie_full', 'gie_gie1', 'gie_gie2', 'gtrans_dw', 'gtrans_sw',
                           'hake_both', 'hake_modonly', 'harotate', 'hole', 'hopfe', 'hyperkg_euclidean',
                           'hyperkg_mobius', 'itransf', 'kg2e_kldivergence', 'kg2e_elikelihood', 'lineare',
                           'lpptransd', 'lpptranse', 'lpptransr', 'makr', 'manifolde_hyperplane', 'manifolde_sphere',
                           'mde', 'mrotate', 'murp_murp', 'murp_mure', 'nage_so3', 'nage_su2', 'pairre',
                           'proje_pointwise', 'proje_listwise', 'protate', 'quatde', 'quate', 'rate', 'reflecte_s',
                           'reflecte_b', 'reflecte_m', 'reflecte_full', 'rode_similarity', 'rode_distance', 'rotate',
                           'rotate3d', 'rotatect', 'rotpro', 'se', 'seek', 'simple_both', 'simple_ignr', 'space',
                           'stranse', 'structure', 'time', 'toruse_L1', 'toruse_L2', 'toruse_eL2', 'trans4e',
                           'transat', 'transcomplex', 'transd', 'transdr', 'transe', 'transedge_cc', 'transedge_cp',
                           'transedt', 'transeft', 'transers', 'transg', 'transgate_fc', 'transgate_wv', 'transh',
                           'transhft', 'transhrs', 'transm', 'transms', 'transmvg', 'transr', 'transrdt', 'transrft',
                           'transsparse_share', 'transsparse_separate', 'transsparsedt_share', 'transsparsedt_separate',
                           'tucker']:
            for dataset in range(0, 10):
                f.write(model_name)
                f.write(',')
                f.write(str(dataset))
                # Change split for NELL!
                if dataset == 3:
                    f.write(',resplit_')
                else:
                    f.write(', ')
                # These are the models that do not have enough hyperparameters.
                if model_name in ['analogy', 'atth_atth', 'atth_roth', 'atth_refh', 'atth_atte', 'atth_rote',
                                  'atth_refe', 'gie_full', 'gie_gie1', 'gie_gie2', 'murp_murp', 'murp_mure', 'tucker']:
                    f.write(',user_bern')
                elif model_name in ['trans4e']:
                        f.write(',user_bern;pnorm')
                else:
                    f.write(',')
                f.write('\n')

    # Missing (NNs): ConvE, HypER, SAttLE
    # Other missing: GCOTE, RESCAL


if __name__ == '__main__':
    run()