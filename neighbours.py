import operator

def euclidean_distance(vector1, vector2):
    return np.sqrt(np.sum(np.power(vector1-vector2, 2)))

def get_neighbours(trainData, testInstance, k):
    distances = []
    neighbors = []
    for idx in range(trainData.shape[0]):
        dist = euclidean_distance(trainData[idx], testInstance)
        distances.append((idx, dist))
    distances.sort(key=operator.itemgetter(1))
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors

def predictkNNClass(output, trainLabels):
    classVotes = {}
    for i in range(len(output)):
        if trainLabels[output[i]] in classVotes:
            classVotes[trainLabels[output[i]]] += 1
        else:
            classVotes[trainLabels[output[i]]] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]


def prediction_accuracy(predicted_labels, original_labels):
    count = 0
    for i in range(len(predicted_labels)):
        if predicted_labels[i] == original_labels[i]:
            count += 1
    return float(count)/len(predicted_labels)


k_vals = [1,3,7]
predicted_classes = {}
final_accuracies = {}

for k in k_vals:
    output_classes = []
    print('Predicting labels for all test samples when k = ' + str(k) + '. Please wait ...')
    for i in range(test_data.shape[0]):
        output = get_neighbours(train_data, test_data[i], k)
        predictedClass = int(predictkNNClass(output, train_labels))
        output_classes.append(predictedClass)
    predicted_classes[k] = output_classes
    final_accuracies[k] = prediction_accuracy(predicted_classes[k], test_labels)
    print('kNN accuracy when k = ' + str(k) + ' is ' + str(final_accuracies[k]))
    print(' ')