import My_Pack as mp
import glob
import os


def main():
    import cv2

    # imgfiles = ["test_1000_6000.tif", "test_11000_10000.tif", "test_2000_0.tif"]
    # imgfiles = ["2.jpg"]

    working_dir = 'New Image Set/'
    ext = 'jpg'
    imgfiles = Get_Files(working_dir, ext)

    testing_set = mp.load_data(imgfiles, working_dir)
    training_set = testing_set

    # Lets Set Params of Train/Test
    # Using Total Detected. Lets do 50/30/20 : Train/Test/Validate
    File_Num = len(imgfiles)
    Train_Num = int(File_Num * 0.5)
    Test_Num = int(File_Num * 0.3)

    training_set = training_set[0:Train_Num]
    testing_set = testing_set[Train_Num:Train_Num+Test_Num]


    # testing_set = testing_set[0:2]

    print(f"Training Set is of {len(training_set)}:")
    # mp.print_filenames(training_set)
    print(f"Testing Set is of {len(testing_set)}:")
    # mp.print_filenames(testing_set)

    # my_obj = mp.LSQ()
    # my_obj.train(training_set)
    # my_obj.validate(testing_set,0)

    my_keras = mp.Keras()
    # # my_keras.load("WholeSet")
    # my_keras.train(training_set)
    my_keras.load("NewImages")
    # my_keras.load_given(training_set)
    # my_keras.Get_Model()
    # my_keras.save("NewImages")
    my_keras.validate(testing_set)

    print("Exiting")
    return 0


def Get_Files(dir, ext):
    owd = os.getcwd()
    os.chdir(dir)
    files = []
    for file in glob.glob("*." + ext):
        # base, _ = os.path.splitext(file)
        files.append(file)
    print(f"Found {len(files)} files to parse")
    os.chdir(owd)
    return files


if __name__ == "__main__":
    main()
