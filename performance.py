import numpy as np
import torch

from sklearn.preprocessing import label_binarize
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                             average_precision_score, precision_recall_curve,
                             balanced_accuracy_score,
                             precision_score, recall_score)

from vis_utils import (draw_cam_on_image, image_to_tensor, tensor_to_image,
                       plot_to_image, get_roc_plot, get_pr_plot,
                       get_conf_mat_plot, get_prediction_plot)

from conv_net_vis.visualisation import GradCam
from conv_net_vis.visualisation.utils import tensor2img


class PerformanceMetrics:
    def __init__(self, true_labels, pred_probs, pred_labels, n_classes, sample_weight):

        self.true_labels = true_labels
        self.pred_probs = pred_probs
        self.pred_labels = pred_labels
        self.n_classes = n_classes
        self.sample_weight = sample_weight

        self.fpr, self.tpr, self.roc_auc = dict(), dict(), dict()
        self.avg_prec = dict()
        self.precision_thr, self.recall_thr = dict(), dict()

        self.weighted_metrics = dict()

    def compute_weighted_metrics(self):

        trues, preds = self.true_labels, self.pred_labels
        weights = [self.sample_weight[int(x)] for x in trues]

        self.precision = precision_score(trues, preds, average='weighted', sample_weight=weights)
        self.recall = recall_score(trues, preds, average='weighted', sample_weight=weights)
        self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)
        self.bal_acc = balanced_accuracy_score(trues, preds, sample_weight=weights)


    def compute_classwise_metrics(self):

        if self.n_classes == 2:
            true_labels = [[1-label, label] for label in self.true_labels]
            true_labels = np.array(true_labels)

            pred_labels = [[1-label, label] for label in self.pred_labels]
            pred_labels = np.array(pred_labels)
        else:
            classes = [x for x in range(self.n_classes)]
            true_labels = label_binarize(self.true_labels, classes=classes)
            pred_labels = label_binarize(self.pred_labels, classes=classes)


        probs = [x.data.cpu().numpy() for x in self.pred_probs]
        probs = np.array(probs)

        for i in range(self.n_classes):

            y_true, y_prob, y_pred = true_labels[:, i], probs[:, i], pred_labels[:, i]
            class_w = [self.sample_weight[int(x)] for x in y_true]

            self.precision_thr[i], self.recall_thr[i], _ = precision_recall_curve(y_true, y_prob, sample_weight=class_w)
            self.avg_prec[i] = average_precision_score(y_true, y_prob, sample_weight=class_w)
            self.fpr[i], self.tpr[i], _ = roc_curve(y_true, y_prob, sample_weight=class_w)
            self.roc_auc[i] = auc(self.fpr[i], self.tpr[i])


class PerformanceTracker:

    def __init__(self, model, cfg, val_loader, summary_writers, epoch):
        self.model = model
        self.cfg = cfg
        self.loader = val_loader
        self.writers = summary_writers
        self.epoch = epoch

        self.n_samples = len(self.loader.dataset)
        self.true_labels = [x[1] for x in self.loader.dataset.data]
        self.pred_labels = []
        self.pred_probs = []

        self.takes = {
            'cam': 4,
            'pred': 16,
        }

        self.grad_input_num = self.takes['cam']
        self.grad_input = []

        self.perf_metrics = None


    def drop_visuals_to_tboard(self):
        self.model.eval()
        self.compute_performance_metrics()
        self.drop_scalars()
        self.drop_pred_images()
        self.drop_cam_images()
        self.drop_conf_mat_images()
        self.drop_pr_image()
        self.drop_roc_image()
        self.model.train()


    def forward(self):

        vis_taken = 0
        self.model.eval()

        with torch.no_grad():
            for batch_ind, (input, classes) in enumerate(self.loader):
                input = input.to(device=self.cfg.DEVICE)

                # forward
                with torch.cuda.amp.autocast():
                    output = self.model(input)

                m = torch.nn.LogSoftmax(dim=0)
                class_probs = []
                for el in output:
                    class_probs += [m(el)]

                _, class_preds = torch.max(output, 1)

                self.pred_probs += class_probs
                self.pred_labels += class_preds.tolist()

                for input_image in input:
                    if vis_taken < self.grad_input_num:
                        vis_taken += 1
                        self.grad_input += [input_image]

        self.model.train()


    def compute_performance_metrics(self):
        self.perf_metrics = PerformanceMetrics(true_labels=self.true_labels,
                                               pred_probs=self.pred_probs,
                                               pred_labels=self.pred_labels,
                                               n_classes=self.cfg.num_classes,
                                               sample_weight=self.cfg.class_weights)
        self.perf_metrics.compute_weighted_metrics()
        self.perf_metrics.compute_classwise_metrics()


    def drop_scalars(self):
        w = self.writers['scalar']
        pm = self.perf_metrics

        for i, class_name in enumerate(self.cfg.class_names):
            w.add_scalar(f'ROC AUC / {class_name}', pm.roc_auc[i], self.epoch)
            w.add_scalar(f'Avg Precision score / {class_name}', pm.avg_prec[i], self.epoch)

        w.add_scalar(f'Performance / Balanced Accuracy score', pm.bal_acc, self.epoch)
        w.add_scalar(f'Performance / Precision score', pm.precision, self.epoch)
        w.add_scalar(f'Performance / Recall score', pm.recall, self.epoch)
        w.add_scalar(f'Performance / F1 score', pm.f1_score, self.epoch)


    def check_accuracy(self, epoch):
        n_correct = 0

        for y, gt in zip(self.pred_labels, self.true_labels):
            is_correct = y == gt
            n_correct += is_correct

        acc = float(n_correct / self.n_samples) * 100
        print(f'End of epoch {epoch}: Guessed {n_correct} / {self.n_samples} with accuracy {acc:.2f}')

        tboard_writer = self.writers['scalar']
        tboard_writer.add_scalar('Training / Accuracy', acc, self.epoch)


    def drop_pred_images(self):
        n_pred_im = self.takes['pred']
        images = [self.loader.dataset.load_image(i) for i in range(n_pred_im)]
        # images = [tensor_to_image(self.loader.dataset[i][0]) for i in range(n_pred_im)]

        figure = get_prediction_plot(
            self.cfg, images, self.true_labels, self.pred_labels)

        preds_image = plot_to_image(figure)
        # cv2.imwrite('prediction.jpg', preds_image)

        tboard_writer = self.writers['predicts']
        tboard_writer.add_image('Prediction', image_to_tensor(preds_image), self.epoch)


    def drop_conf_mat_images(self):

        cm = confusion_matrix(self.true_labels, self.pred_labels)
        figure = get_conf_mat_plot(cm, self.cfg.class_names_short)
        cm_image = plot_to_image(figure)
        # cv2.imwrite('confusion_mat.jpg', cm_image)

        tboard_writer = self.writers['conf_mat']
        tboard_writer.add_image('Confusion matrix', image_to_tensor(cm_image), self.epoch)


    def drop_cam_images(self):

        n = self.grad_input_num
        w, h = self.cfg.IMG_WIDTH, self.cfg.IMG_HEIGHT

        images = [self.loader.dataset.load_image(i)
                  for i in range(n)]
        combined = np.zeros((h * 2, w * n, 3), dtype=np.uint8)

        vis = GradCam(self.model, self.cfg.DEVICE)
    
        for i, (image, tensor) in enumerate(zip(images, self.grad_input)):

            model_outs = vis(torch.unsqueeze(tensor, 0), None)[0]
            cam = tensor2img(model_outs[0])

            img_with_cam = draw_cam_on_image(image, cam)
            im_and_cam = np.vstack((image, img_with_cam))
            combined[:, w *i:w *(i+1), :] = im_and_cam

        # cv2.imwrite('cam.jpg', combined)
        tboard_writer = self.writers['gradcam']
        tboard_writer.add_image('Class Activation Map', image_to_tensor(combined), self.epoch)
        

    def drop_pr_image(self):

        figure_pr = get_pr_plot(self.perf_metrics.precision_thr,
                                self.perf_metrics.recall_thr,
                                self.perf_metrics.avg_prec,
                                self.cfg.class_names)
        pr_image = plot_to_image(figure_pr)
        # cv2.imwrite('pr_curve_image.jpg', pr_image)

        tboard_writer = self.writers['pr_curve']
        tboard_writer.add_image('Precision-Recall curve', image_to_tensor(pr_image), self.epoch)


    def drop_roc_image(self):

        roc_figure = get_roc_plot(self.perf_metrics.fpr,
                                  self.perf_metrics. tpr,
                                  self.perf_metrics.roc_auc,
                                  self.cfg.class_names)
        roc_image = plot_to_image(roc_figure)
        # cv2.imwrite('roc_image.jpg', roc_image)

        tboard_writer = self.writers['roc_curve']
        tboard_writer.add_image('Receiver operating characteristic',
                                image_to_tensor(roc_image), self.epoch)

