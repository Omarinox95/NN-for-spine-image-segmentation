
from os import walk
import nibabel as nib


def extract_and_match_data_walk(dirpath, ending, ending_2, ending_struc):

    foldername_saver = []
    foldername_saver_2 = []
    foldername_saver_struc = []

    for path, subfolders, files in walk(dirpath):

        for file in files:
            if file.endswith(ending):
                foldername_saver.append(path+"/"+file)  # use the folder direction

            elif file.endswith(ending_2):
                foldername_saver_2.append(path+"/"+file)

            elif file.endswith(ending_struc):
                foldername_saver_struc.append(path + "/" + file)

    return foldername_saver, foldername_saver_2, foldername_saver_struc

