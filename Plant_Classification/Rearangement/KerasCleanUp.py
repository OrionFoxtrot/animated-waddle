import My_Package as mp
import numpy as np

def main():
    print("Hello CleanUp")
    imgfile = "test_1000_6000.tif"


    img,x,y = mp.Read_Image(imgfile) # Read
    pos_mask,neg_mask = mp.Generate_Masks(x,y) # Masks
    sample_weight,any_mask,labels = mp.Generate_Weights(img,pos_mask,neg_mask) #Weights

    h, w = img.shape[:2]
    classifier_only, full_model = mp.create_models(h, w)

    # images = np.stack((img1, img2, ...))
    # weights = np.stack((weights1, weights2, ...))
    # labels = ...

    # full_model.fit(x=(images, weights, labels))

    full_model.fit(x=(img[None, :], sample_weight[None, :], labels[None, :]),
                   epochs=60)

    weights, bias = full_model.get_layer('my_dense').get_weights()

    print('weights:', weights)
    print('bias:', bias)

    pred_probs = classifier_only.predict(img[None, :])
    pred_probs = pred_probs[0]  # get rid of 'batch' dimension

    pred_labels = np.round(pred_probs)  # 1's and 0's

    correct = (labels[any_mask] == pred_labels[any_mask])

    weighted_accuracy = (correct * sample_weight[any_mask]).sum()
    unweighted_accuracy = correct.mean()

    print('weighted accuracy on just masked pixels:', weighted_accuracy)
    print('unweighted accuracy on just masked pixels:', unweighted_accuracy)

    #Evaluation
    imgfile = "test_11000_10000.tif"
    Eval_img, Eval_x, Eval_y = mp.Read_Image(imgfile)
    Eval_pos_mask,Eval_neg_mask = mp.Generate_Masks(Eval_x,Eval_y)
    Eval_sample_weight, Eval_any_mask, Eval_labels = mp.Generate_Weights(Eval_img, Eval_pos_mask, Eval_neg_mask)

    results= full_model.evaluate(x=(Eval_img[None, :], Eval_sample_weight[None, :], Eval_labels[None, :]),
                   batch_size=1)
    print("test loss, test accuracy:", results)


    full_model.save('MyModel')
    classifier_only.save('MyClassifier')



if __name__ == '__main__':
    main()