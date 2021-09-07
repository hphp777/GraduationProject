from .classifier_model import *

selected_labels = np.take(labels, np.argsort(auc_rocs_after)[::-1])[:4]

def gradcam(img, img_dir):
    compute_gradcam(
        model,
        img,
        img_dir,
        train_df_main,
        labels,
        selected_labels,
        layer_name="efficientnet-b1",
    )