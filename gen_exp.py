def run():
    filename = 'Exp1.txt'

    with open(filename, 'w') as f:
        for model_name in ['transe', 'analogy', 'aprile', 'boxe',
                           'combine', 'complex', 'crosse', 'crosses', 'cycle']:
            for dataset in range(0, 8):
                f.write(model_name)
                f.write(',')
                f.write(str(dataset))
                f.write(', ')
                f.write(',0')
                f.write('\n')


if __name__ == '__main__':
    run()