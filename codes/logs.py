import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-path', default=None, type=str, help='logs file path')
    args = parser.parse_args()
    file1 = open(args.path, 'r')
    Lines = file1.readlines()
    print(Lines[-1])
