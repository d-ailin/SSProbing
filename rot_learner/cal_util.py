'''
    https://github.com/markus93/NN_calibration/blob/eb235cdba006882d74a87114a3563a9efca691b7/scripts/utility/evaluation.py
    https://github.com/markus93/NN_calibration/blob/master/scripts/calibration/cal_methods.py
'''
import numpy as np
from scipy.optimize import minimize 
from sklearn.metrics import log_loss
import pandas as pd
import time
from sklearn.metrics import log_loss, brier_score_loss
import sklearn.metrics as metrics
import sys
from os import path

import torch
import torch.nn.functional as F
# Imports to get "utility" package
sys.path.append( path.dirname( path.dirname( path.abspath("utility") ) ) )
# from utility.unpickle_probs import unpickle_probs
# from utility.evaluation import ECE, MCE


def _compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, correct):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(correct, conf) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == 1])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[1] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def compute_acc_bin(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin
    
    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
    
    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [x for x in zip(pred, true, conf) if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper]
    if len(filtered_tuples) < 1:
        return 0,0,0
    else:
        correct = len([x for x in filtered_tuples if x[0] == x[1]])  # How many correct labels
        len_bin = len(filtered_tuples)  # How many elements falls into given bin
        avg_conf = sum([x[2] for x in filtered_tuples]) / len_bin  # Avg confidence of BIN
        accuracy = float(correct)/len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin

def eval_ece(conf, corrects, bin_size = .1):
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = _compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, corrects)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece

def eval_mce(conf, corrects, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = _compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, corrects)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)

def ECE(conf, pred, true, bin_size = 0.1):

    """
    Expected Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        ece: expected calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)  # Get bounds of bins
    
    n = len(conf)
    ece = 0  # Starting error
    
    for conf_thresh in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)        
        ece += np.abs(acc-avg_conf)*len_bin/n  # Add weigthed difference to ECE
        
    return ece
        
      
def MCE(conf, pred, true, bin_size = 0.1):

    """
    Maximal Calibration Error
    
    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?
        
    Returns:
        mce: maximum calibration error
    """
    
    upper_bounds = np.arange(bin_size, 1+bin_size, bin_size)
    
    cal_errors = []
    
    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(conf_thresh-bin_size, conf_thresh, conf, pred, true)
        cal_errors.append(np.abs(acc-avg_conf))
        
    return max(cal_errors)
    

def evaluate(probs, y_true, verbose = False, normalize = False, bins = 15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, MCE, NLL, Brier Score
    
    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)
        
    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """
    
    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    
    if normalize:
        confs = np.max(probs, axis=1)/np.sum(probs, axis=1)
        # Check if everything below or equal to 1?
    else:
        confs = np.max(probs, axis=1)  # Take only maximum confidence
    
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy
    
        # Calculate ECE
    ece = ECE(confs, preds, y_true, bin_size = 1/bins)
    # Calculate MCE
    mce = MCE(confs, preds, y_true, bin_size = 1/bins)
    
    loss = log_loss(y_true=y_true, y_pred=probs)
    
    y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class

    y_true_onehot = np.zeros((y_true.size, y_true.max() + 1))
    y_true_onehot[np.arange(y_true.size), y_true] = 1

    brier = brier_score_loss(y_true_onehot.flatten(), probs.flatten())

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("MCE:", mce)
        print("Loss:", loss)
        print("brier:", brier)
    
    return (error, ece, mce, loss, brier)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.
    
    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """

    # old
    # e_x = np.exp(x - np.max(x))
    # # e_x = np.exp(x)
    # out = e_x / e_x.sum(axis=1, keepdims=1)
    # # if np.isnan(out).sum() > 0:
    # #     print('has nan:', np.isnan(out).sum())
    # #     exit()

    # return out

    # new
    x_ts = torch.tensor(x)
    return F.softmax(x_ts, dim=1).numpy()
    

class HistogramBinning():
    """
    Histogram Binning as a calibration method. The bins are divided into equal lengths.
    
    The class contains two methods:
        - fit(probs, true), that should be used with validation data to train the calibration model.
        - predict(probs), this method is used to calibrate the confidences.
    """
    
    def __init__(self, M=15):
        """
        M (int): the number of equal-length bins used
        """
        self.bin_size = 1./M  # Calculate bin size
        self.conf = []  # Initiate confidence list
        self.upper_bounds = np.arange(self.bin_size, 1+self.bin_size, self.bin_size)  # Set bin bounds for intervals

    
    def _get_conf(self, conf_thresh_lower, conf_thresh_upper, probs, true):
        """
        Inner method to calculate optimal confidence for certain probability range
        
        Params:
            - conf_thresh_lower (float): start of the interval (not included)
            - conf_thresh_upper (float): end of the interval (included)
            - probs : list of probabilities.
            - true : list with true labels, where 1 is positive class and 0 is negative).
        """

        # Filter labels within probability range
        filtered = [x[0] for x in zip(true, probs) if x[1] > conf_thresh_lower and x[1] <= conf_thresh_upper]
        nr_elems = len(filtered)  # Number of elements in the list.

        if nr_elems < 1:
            return 0
        else:
            # In essence the confidence equals to the average accuracy of a bin
            conf = sum(filtered)/nr_elems  # Sums positive classes
            return conf
    

    def fit(self, probs, true):
        """
        Fit the calibration model, finding optimal confidences for all the bins.
        
        Params:
            probs: probabilities of data
            true: true labels of data
        """

        conf = []

        # Got through intervals and add confidence to list
        for conf_thresh in self.upper_bounds:
            temp_conf = self._get_conf((conf_thresh - self.bin_size), conf_thresh, probs = probs, true = true)
            conf.append(temp_conf)

        self.conf = conf


    # Fit based on predicted confidence
    def predict(self, probs):
        """
        Calibrate the confidences
        
        Param:
            probs: probabilities of the data (shape [samples, classes])
            
        Returns:
            Calibrated probabilities (shape [samples, classes])
        """

        # Go through all the probs and check what confidence is suitable for it.
        for i, prob in enumerate(probs):
            idx = np.searchsorted(self.upper_bounds, prob)
            probs[i] = self.conf[idx]

        return probs
        
        
class TemperatureScaling():
    
    def __init__(self, temp = 1, maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        if np.isnan(loss).sum() > 0:
            print('np.isnan(loss).sum()', np.isnan(loss).sum())
        return loss
    
    # Find the temperature
    def fit(self, logits, true):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        opt = minimize(self._loss_fun, x0 = 1, args=(logits, true), options={'maxiter':self.maxiter}, method = self.solver)
        self.temp = opt.x[0]
        print('finished temp', self.temp)
        
        return opt
        
    def predict(self, logits, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if not temp:
            return softmax(logits/self.temp)
        else:
            return softmax(logits/temp)


class TemperatureScalingWithSSL():
    
    def __init__(self, temp = np.ones(3), maxiter = 50, solver = "BFGS"):
        """
        Initialize class
        
        Params:
            temp (float): starting temperature, default 1
            maxiter (int): maximum iterations done by optimizer, however 8 iterations have been maximum.
        """
        self.temp = temp
        self.maxiter = maxiter
        self.solver = solver
    
    def _loss_fun(self, x, probs, true, ssl):
        # Calculates the loss using log-loss (cross-entropy loss)
        scaled_probs = self.predict(probs, ssl, x)
        # if (scaled_probs == 0).sum() > 0:
        #     print('(scaled_probs == 0).sum()', (scaled_probs == 0).sum())
        # if np.isnan(scaled_probs).sum() > 0:
        #     print('np.isnan(scaled_probs).sum()', np.isnan(scaled_probs).sum())

        # loss = log_loss(y_true=true, y_pred=scaled_probs)
        # print('current x', x)
        loss = log_loss(y_true=true, y_pred=scaled_probs)
        # loss = F.cross_entropy( torch.Tensor(scaled_probs), torch.Tensor(true).long() ).numpy()
        # if np.isnan(loss).sum() > 0:
        #     print('np.isnan(loss).sum()', np.isnan(loss).sum())
        return loss
    
    # Find the temperature
    def fit(self, logits, true, ssl):
        """
        Trains the model and finds optimal temperature
        
        Params:
            logits: the output from neural network for each class (shape [samples, classes])
            true: one-hot-encoding of true labels.
            
        Returns:
            the results of optimizer after minimizing is finished.
        """
        
        true = true.flatten() # Flatten y_val
        # opt = minimize(self._loss_fun, x0 = np.ones(3), args=(logits, true, ssl), options={'maxiter':self.maxiter, 'disp':True}, method = self.solver, bounds=[(-5, 5), (-5, 5), (-5, 5)])
        # need to initilize with very small value close to 0, otherwise will get trival solution or get NaN when having too many points.
        # ok for cifar10 resnet
        # opt = minimize(self._loss_fun, x0 = np.ones(3)/10, args=(logits, true, ssl), options={'maxiter':self.maxiter, 'disp':True}, method = self.solver, bounds=[(-5, 5), (-5, 5), (-5, 5)])
        
        fit_nums = ssl.shape[1] + 1 # ssl and original

        # debug for cinic10
        # opt = minimize(self._loss_fun, x0 = np.ones(fit_nums)/2, args=(logits, true, ssl), options={'maxiter':self.maxiter, 'disp':True}, method = self.solver)
        opt = minimize(self._loss_fun, x0 = np.ones(fit_nums)/2, args=(logits, true, ssl), options={'maxiter':self.maxiter}, method = self.solver)
        # self.temp = opt.x[0]
        self.temp = opt.x
        # print('finished temp', self.temp)
        
        return opt
        
    def predict(self, logits, ssl, temp = None):
        """
        Scales logits based on the temperature and returns calibrated probabilities
        
        Params:
            logits: logits values of data (output from neural network) for each class (shape [samples, classes])
            temp: if not set use temperatures find by model or previously set.
            
        Returns:
            calibrated probabilities (nd.array with shape [samples, classes])
        """
        
        if temp is None:
            temp = self.temp

        temp_ssl = (temp[0] - temp[1] * ssl[:, 0] - temp[2] * ssl[:, 1]).reshape(-1, 1)
            # return softmax(logits/temp)
        # add epilson
        # temp_ssl[temp_ssl == 0] = 1e-8
        # return softmax(logits/(temp_ssl))

        # or change to mulitple
        return softmax(logits * temp_ssl)


def cal_results(fn, res, m_kwargs = {}, approach = "all"):
    
    """
    Calibrate models scores, using output from logits files and given function (fn). 
    There are implemented to different approaches "all" and "1-vs-K" for calibration,
    the approach of calibration should match with function used for calibration.
    
    TODO: split calibration of single and all into separate functions for more use cases.
    
    Params:
        fn (class): class of the calibration method used. It must contain methods "fit" and "predict", 
                    where first fits the models and second outputs calibrated probabilities.
        path (string): path to the folder with logits files
        files (list of strings): pickled logits files ((logits_val, y_val), (logits_test, y_test))
        m_kwargs (dictionary): keyword arguments for the calibration class initialization
        approach (string): "all" for multiclass calibration and "1-vs-K" for 1-vs-K approach.
        
    Returns:
        df (pandas.DataFrame): dataframe with calibrated and uncalibrated results for all the input files.
    
    """
    
    df = pd.DataFrame(columns=["Name", "Error", "ECE", "MCE", "Loss", "Brier"])
    
    total_t1 = time.time()
    
    # for i, f in enumerate(files):
        
        # name = "_".join(f.split("_")[1:-1])
        # print(name)
    t1 = time.time()

    
    (logits_val, y_val), (logits_test, y_test) = res
    # FILE_PATH = join(path, f)
    # (logits_val, y_val), (logits_test, y_test) = unpickle_probs(FILE_PATH)
    
    if approach == "all":            

        y_val = y_val.flatten()

        model = fn(**m_kwargs)

        model.fit(logits_val, y_val)

        probs_val = model.predict(logits_val) 
        probs_test = model.predict(logits_test)
        
        # error, ece, mce, loss, brier = evaluate(softmax(logits_test), y_test, verbose=True)  # Test before scaling
        error, ece, mce, loss, brier = evaluate(softmax(logits_test), y_test, verbose=False)  # Test before scaling
        error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False)
        # error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=True)
        
        # print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False, normalize=True))
        # print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_test, y_test, verbose=False, normalize=True))

        print('Validation')
        print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False, normalize=True))
        print('Test')
        print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_test, y_test, verbose=False, normalize=True))

        
    else:  # 1-vs-k models
        probs_val = softmax(logits_val)  # Softmax logits
        probs_test = softmax(logits_test)
        K = probs_test.shape[1]
        
        # Go through all the classes
        for k in range(K):
            # Prep class labels (1 fixed true class, 0 other classes)
            y_cal = np.array(y_val.reshape(-1, 1) == k, dtype="int")[:, 0]

            # Train model
            model = fn(**m_kwargs)
            model.fit(probs_val[:, k], y_cal) # Get only one column with probs for given class "k"

            probs_val[:, k] = model.predict(probs_val[:, k])  # Predict new values based on the fittting
            probs_test[:, k] = model.predict(probs_test[:, k])

            # Replace NaN with 0, as it should be close to zero  # TODO is it needed?
            idx_nan = np.where(np.isnan(probs_test))
            probs_test[idx_nan] = 0

            idx_nan = np.where(np.isnan(probs_val))
            probs_val[idx_nan] = 0

        # Get results for test set
        error, ece, mce, loss, brier = evaluate(softmax(logits_test), y_test, verbose=False, normalize=False)
        error2, ece2, mce2, loss2, brier2 = evaluate(probs_test, y_test, verbose=False, normalize=True)
        
        print('Validation')
        print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_val, y_val, verbose=False, normalize=True))
        print('Test')
        print("Error %f; ece %f; mce %f; loss %f, brier %f" % evaluate(probs_test, y_test, verbose=False, normalize=True))
        
    
    # df.loc[i*2] = [name, error, ece, mce, loss, brier]
    # df.loc[i*2+1] = [(name + "_calib"), error2, ece2, mce2, loss2, brier2]
    
    # t2 = time.time()
    # print("Time taken:", (t2-t1), "\n")
        
    # total_t2 = time.time()
    # print("Total time taken:", (total_t2-total_t1))
        
    return evaluate(probs_test, y_test, verbose=False, normalize=True)
 