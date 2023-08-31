from scipy.stats import qmc


def run():
    # s = ''
    # last_init, last_end = 34, 50
    # for i in range(19):
    #     s += str(last_init)+'-'+str(last_end)+','
    #
    #     #for j in range(last_init, last_end+1):
    #     #    print('scancel 15984621_'+str(j))
    #
    #     last_init += 153
    #     last_end += 153
    # print(s)

    good_samples = 0
    while good_samples < 2:
        good_samples = 0
        # Generate Sobol sequence.
        sampler = qmc.Sobol(d=4, scramble=True)
        sample = sampler.random_base2(m=3)

        sample /= 2
        # Check the results:
        for s in sample:
            if s[0] > s[1] > s[2] and s[0] > s[3] > s[1] and s[3] > s[2]:
                good_samples += 1

    with open('Calib', 'w') as f:
        for dataset in [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]:
            for calib_type in ['isotonic', 'platt']:
                for model_name in ['boxe', 'complex', 'hake_both', 'hole', 'quate', 'rotate', 'rotpro',
                                   'toruse_eL2', 'transe', 'tucker']:
                    for i in range(1+len(sample)):
                        if i == 0:
                            # Our approach.
                            #targets = [0.45, 0.15, 0, 0.25]
                        #elif i == 1:
                            # Original approach.
                            targets = [0, 0, 0, 0]
                        else:
                            targets = sample[i-1]

                        f.write(model_name)
                        f.write(',')
                        f.write(str(dataset))
                        if dataset == 3:
                            f.write(',resplit_')
                        else:
                            f.write(', ')
                        f.write(',0')
                        f.write(',')
                        f.write(calib_type)
                        f.write(',1,')
                        f.write(",".join([str(t) for t in targets]))
                        f.write('\n')


if __name__ == '__main__':
    run()