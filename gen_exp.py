def run():
    filename = 'Expl'

    with open(filename, 'w') as f:
        for model_name in ['boxe', 'complex', 'hake_both', 'hole', 'quate', 'rotate', 'rotpro', 'toruse_eL2', 'transe',
                           'tucker']:
            for dataset in range(0, 8):
                f.write(model_name)
                f.write(',')
                f.write(str(dataset))
                f.write(', ')
                f.write(',0')
                f.write('\n')


if __name__ == '__main__':
    run()