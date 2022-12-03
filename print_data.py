import numpy as np

def main():
    path = "./patchnetvlad/dataset_gt_files/tokyo247.npz"

    data = np.load(path)
    
    query = data["utmQ"]
    database = data["utmDb"]
    thresholds = data["posDistThr"]

    print(query.shape)
    print(query[:5])
    print()

    print(database.shape)
    print(database[:5])
    print()

    print(thresholds)

if __name__ == "__main__":
    main()
