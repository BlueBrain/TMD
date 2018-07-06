list_of_modules = ['discriminant_analysis', 'discriminant_analysis', 'tree']

list_of_classifiers = ['LinearDiscriminantAnalysis',
                       'QuadraticDiscriminantAnalysis', 'DecisionTreeClassifier']


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


def leave_one_out(mod, classifier, data, labels, **kwargs):
    '''Leaves one individual out, trains classifier
       with the rest of the data and returns the score
       of matching ids between proposed and predicted labels.
       Score defines how many trials were successful
       as a percentage over the total number of trials.
    '''
    sample_size = len(labels)
    scores = np.zeros(sample_size)

    for ed in range(sample_size):

        # print 'Testing ' + str(ed) + ' ...'

        train_data = data[np.delete(range(sample_size), ed)]
        train_labels = labels[np.delete(range(sample_size), ed)]

        clf = train(mod, classifier, train_data, train_labels, **kwargs)
        predict_label = predict(clf, data[ed])

        # print 'The individual ' + str(ed) + ' is of type ' + str(predict_label)

        scores[ed] = predict_label == labels[ed]

    return np.float(np.count_nonzero(scores)) / sample_size


def leave_one_out_statistics(mod, classifier, data, labels, N=10, **kwargs):
    '''Leaves one individual out, trains classifier
       with the rest of the data and returns the score
       of matching ids between proposed and predicted labels.
       Score defines how many trials were successful
       as a percentage over the total number of trials.
    '''
    sample_size = len(labels)
    scores = np.zeros(sample_size)

    for ed in range(sample_size):

        # print 'Testing ' + str(ed) + ' ...'

        train_data = data[np.delete(range(sample_size), ed)]
        train_labels = labels[np.delete(range(sample_size), ed)]

        clf = train(mod, classifier, train_data, train_labels, **kwargs)

        all_results = []
        for i in range(N):
            all_results.append(predict(clf, data[ed]))

        predict_label = predict(clf, data[ed])
        print 'The individual ', str(ed), ' is of type ', all_results

        scores[ed] = predict_label == labels[ed]

    return np.float(np.count_nonzero(scores)) / sample_size


def leave_perc_out(mod, classifier, data, labels, iterations=10, percent=10, **kwargs):
    '''Leaves one individual out, trains classifier
       with the rest of the data and returns the score
       of matching ids between proposed and predicted labels.
       Score defines how many trials were successful
       as a percentage over the total number of trials.
       Iteration defines the number of trials.
       Percent defines the percentage of the data that will
       define the test set of the classifier.
    '''
    import random

    sample_size = len(labels)
    test_size = int(sample_size * percent / 100.)
    scores = np.zeros(iterations)

    # print sample_size, test_size

    for i in range(iterations):

        random_inds = random.sample(range(0, sample_size), test_size)
        kept = np.delete(range(sample_size), random_inds)

        clf = train(mod, classifier, data[kept], labels[kept], **kwargs)

        sc = 0.0
        for ed in random_inds:
            predict_label = predict(clf, data[ed])
            sc = sc + float(predict_label == labels[ed])

        scores[i] = float(sc) / float(test_size)

    # print len(random_inds), len(kept), len(random_inds) + len(kept)

    return scores  # np.mean(np.count_nonzero(scores))/sample_size


def leave_one_out_mixing(mod, classifier, data, labels, **kwargs):
    '''Leaves one individual out, trains classifier
       with the rest of the data and returns the score
       of matching ids between proposed and predicted labels.
       Score defines how many trials were successful
       as a percentage over the total number of trials.
    '''
    sample_size = len(labels)
    scores = np.zeros(sample_size)

    separation = np.zeros([len(np.unique(labels)), len(np.unique(labels))])

    sizes = np.zeros(len(np.unique(labels)))

    for i in np.unique(labels):
        sizes[int(i - 1)] = len(np.where(labels == i)[0])

    for ed in range(sample_size):

        # print 'Testing ' + str(ed) + ' ...'

        train_data = data[np.delete(range(sample_size), ed)]
        train_labels = labels[np.delete(range(sample_size), ed)]

        clf = train(mod, classifier, train_data, train_labels, **kwargs)
        predict_label = predict(clf, data[ed])

        # print predict_label, labels[ed]
        separation[int(labels[ed] - 1)][int(predict_label - 1)] = separation[int(labels[ed] - 1)
                                                                             ][int(predict_label - 1)] + 1. / sizes[int(labels[ed] - 1)]

        # print 'The individual ' + str(ed) + ' is of type ' + str(predict_label)

        scores[ed] = predict_label == labels[ed]

    return np.float(np.count_nonzero(scores)) / sample_size, separation


def leave_one_out_multiple(mod, classifier, data, labels, n=10, **kwargs):
    '''Leaves one individual out, trains classifier
       with the rest of the data and returns the score
       of matching ids between proposed and predicted labels.
       Score defines how many trials were successful
       as a percentage over the total number of trials.
    '''
    sample_size = len(labels)
    scores = np.zeros(sample_size)

    for ed in range(sample_size):

        # print 'Testing ' + str(ed) + ' ...'

        print 'The individual ' + str(ed) + ' is of type ',

        for ni in range(n):

            train_data = data[np.delete(range(sample_size), ed)]
            train_labels = labels[np.delete(range(sample_size), ed)]

            clf = train(mod, classifier, train_data, train_labels, **kwargs)

            predict_label = predict(clf, data[ed])

            print str(predict_label),

        print ' !'

        scores[ed] = predict_label == labels[ed]

    return np.float(np.count_nonzero(scores)) / sample_size


def multi(dat, tar, m='tree', cl='DecisionTreeClassifier', n=10, randomize=False):
    score = np.zeros(n)
    if not randomize:
        for i in range(n):
            score[i] = leave_one_out(m, cl, dat, tar)
    else:
        for i in range(n):
            score[i] = leave_one_out(m, cl, dat, np.random.randint(
                min(tar), max(tar) + 1, size=len(tar)))

    return mean(score), std(score)
