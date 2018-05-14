import tmd
import numpy as np
# import view

list_of_modules = ['discriminant_analysis', 'discriminant_analysis', 'tree']

list_of_classifiers =['LinearDiscriminantAnalysis', 'QuadraticDiscriminantAnalysis', 'DecisionTreeClassifier']

def train(mod, classifier, data, labels, **kwargs):
    '''Trains the classifier from mod of sklearn
       with data and targets.
       Returns a fited classifier.
    '''
    import importlib

    clas_mod = importlib.import_module('sklearn.' + mod)
    clf = getattr(clas_mod, classifier)()
    clf.set_params(**kwargs)

    clf.fit(data, labels)

    return clf

def predict(clf, data):
    '''Predict label for data for the trained classifier clf.
       Returns the index of the predicted class
       for each datapoint in data.
    '''
    predict_label = clf.predict([data])

    return predict_label[0]

def classify_cell_in_groups(list_of_groups=['L5_UTPC', 'L5_STPC', 'L5_TTPC1', 'L5_TTPC2'],
                            cell_to_classify='./L5_TTPC1/C030796A-P3.h5',
                            neurite_type='apical',
                            classifier_module=list_of_modules[0],
                            classifier_method=list_of_classifiers[0],
                            number_of_trials=20):

    # ------------------------ Training dataset --------------------------------
    # Load all data from selected folders
    groups = [tmd.io.load_population(l) for l in list_of_groups]
    # Define labels depending on the number of neurons in each folder
    labels = [i+1 for i,k in enumerate(groups) for j in k.neurons]
    # Generate a persistence diagram per neuron
    pers_diagrams = [tmd.methods.get_ph_neuron(j, neurite_type=neurite_type)
                     for i,k in enumerate(groups) for j in k.neurons]
    # Define x-ylimits
    xlims, ylims = tmd.analysis.define_limits(pers_diagrams)
    # Generate a persistence image for each diagram
    pers_images = [tmd.analysis.persistence_image_data(p, xlims=xlims, ylims=ylims)
                   for p in pers_diagrams]
    # Create the train dataset from the flatten images
    train_dataset = [i.flatten() for i in pers_images]

    # ------------------------ Test cell dataset -------------------------------
    # Load cell to be classified
    neuron2test = tmd.io.load_neuron(cell_to_classify)
    # Get persistence diagram from test cell
    pers2test = tmd.methods.get_ph_neuron(neuron2test, neurite_type=neurite_type)
    # Get persistence image from test cell
    pers_image2test = tmd.analysis.persistence_image_data(pers2test, xlims=xlims, ylims=ylims)
    # Create the test dataset from the flatten image of the test cell
    test_dataset = pers_image2test.flatten()

    predict_labels = []
    # Train classifier with training images for selected number_of_trials
    for i in xrange(number_of_trials):

        clf = train(classifier_module, classifier_method, train_dataset, labels)
        # Test classifier with test image and return predictions
        predict_labels.append(predict(clf, test_dataset))

    percentages = {groups[i-1].name: np.float(len(np.where(np.array(predict_labels) == i)[0])) / len(predict_labels)
                   for i in np.unique(labels)}

    return percentages
