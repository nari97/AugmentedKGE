import sys
import pickle

def run():
    #folder = sys.argv[1]
    folder = ""

    result = {}
    result['trial_index'] = 9
    result_file = folder + "Ax/rescal_3_"+str(result['trial_index'])+".fail"
    with open(result_file, 'wb') as f:
        pickle.dump(result, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    run()