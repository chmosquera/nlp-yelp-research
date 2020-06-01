import sys
import pandas as pd
import pickle
import os.path
from os import path
import time

# args: inputFile, outputFile
# output: raw dataframe, cleaned dataframe
# 1. Open the .csv and save as a pandas dataset object
# 2. clean the text: tokenized, filtered (stop words, punctuation, lowercase, alpha), lemma

def pickleDataframe():
    use_batches = False
    if use_batches:
        batch_num = 1000000
        i = 1       # change this variable, current batch number
        df = pd.read_csv(inputFile, skiprows=(i * batch_num), nrows=((i+1)*batch_num))
        df.to_pickle(picklePath + outputFile)
    else:
        start = time.time()
        df = pd.read_csv(inputFile)
        df.to_pickle(picklePath + outputFile)
        end = time.time()
        print("Time elapsed: ", end - start)

def testPickleDataFrame(input, output):
    print("Testing pickleDataFrame():")
    # open datafile for reading
    df = pd.read_csv(inputFile, nrows=10)
    # save as pickle
    df.to_pickle(picklePath + outputFile)
    print("- Saved file '", input, "' into pickle '", output, "'")
    # load pickle
    data = pd.read_pickle(picklePath + 'yelp_review.pkl')
    print("- Reading first ten lines of pickled dataset:")
    print(data.head(30))

def main():
    global inputFile, outputFile
    if (not len(sys.argv) == 3):
        print("Argument error. usage: python prepare_data.py inputFile, outputFile")
        exit()
    inputFile = sys.argv[1]
    outputFile = sys.argv[2]
    pickleDataframe()

if __name__ == "__main__":
    inputFile = ""
    outputFile = ""
    picklePath = "pickles\\"

    # setup pickle directory if non existent
    if not path.exists(picklePath):
        try:
            os.mkdir(os.getcwd() + "\\" + picklePath)
        except OSError:
            print("Creation of the directory %s failed" % picklePath)
        else:
            print("Successfully created the directory %s " % picklePath)

    # Finally, do main stuff
    main()
