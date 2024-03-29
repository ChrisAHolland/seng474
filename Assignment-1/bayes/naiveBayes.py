'''
Naive Bayes Multinomial Text Classifier

Classifies sentences into 2 classes
Class 0: A message that contains a wise saying
Class 1: A message that predicts the future
'''

'''
Trains a multinomial model.
Partially based on pseudocodegiven in class.
'''
def trainMultinomial():
    with open("traindata.txt", "r") as trainData, open("trainlabels.txt", "r") as trainLabels:
        numClass1 = lines = wordCount1 = wordCount0 = 0
        vocab1 = {}
        vocab0 = {}
        trainDataLines = trainData.readlines()
        for i, line in enumerate(trainLabels):
            lines += 1
            if line.rstrip('\n') == '1':
                numClass1 += 1
                wordCount1 += getVocab(trainDataLines[i], vocab1)
            else:
                wordCount0 += getVocab(trainDataLines[i], vocab0)

        totalVocab = len(set(list(vocab1.keys()) + list(vocab0.keys())))
        probC = numClass1 / lines
        vocabProb1 = getProb(vocab1, totalVocab, wordCount1)
        vocabProb0 = getProb(vocab0, totalVocab, wordCount0)

        return (vocabProb1, vocabProb0, wordCount1, wordCount0, probC, totalVocab)

'''
Apply's a multinomial model to a dataset.
Partially based on pseudocodegiven in class.
'''
def applyMultinomial(line, vocabProb1, vocabProb0, wordCount1, wordCount0, probC, totalVocab):
    prob1 = probC
    prob0 = 1 - prob1

    for word in line.split():
        if word in vocabProb1.keys():
            prob1 *= vocabProb1[word]
        else:
            prob1 *= 1 / (wordCount1 + totalVocab)

        if word in vocabProb0.keys():
            prob0 *= vocabProb0[word]
        else:
            prob0 *= 1 / (wordCount0 + totalVocab)

    if prob1 > prob0:
        return 1
    else:
        return 0

'''
Helper function.
Calculates the probability of a words being in a Document.
'''
def getProb(vocab, total, numWords):
    vocabProb = {}
    for word in vocab:
        prob = (vocab[word] + 1) / (numWords + total)
        vocabProb[word] = prob
    return vocabProb

'''
Helper function.
Extracts the vocab from a Document.
'''
def getVocab(line, vocab):
    wordCount = 0
    wordList = line.split()
    for word in wordList:
        wordCount +=1
        if word in vocab:
            vocab[word] += 1
        else:
            vocab[word] = 1
    return wordCount

def main():
    trained = trainMultinomial()
    vocabProb1 = trained[0]
    vocabProb0 = trained[1]
    wordCount1 = trained[2]
    wordCount0 = trained[3]
    probC = trained[4] # My OCD hates this
    totalVocab = trained[5]

    with open("traindata.txt", "r") as trainData, open("trainlabels.txt", "r") as trainLabels:
        labelLines = trainLabels.readlines()
        result = []
        correct = 0

        for i, line in enumerate(trainData):
            result.append(applyMultinomial(line, vocabProb1, vocabProb0, wordCount1, wordCount0, probC, totalVocab))

        for i in range(0, len(result) - 1):
            if result[i] == int(labelLines[i].strip()):
                correct += 1

        result = correct / len(result)
        f = open('results.txt', 'w')
        f.write("Training Files: traindata.txt, trainlabels.txt\n")
        f.write("Test Files: traindata.txt, trainlabels.txt\n")
        f.write("Result: ")
        f.write(str(result))
        f.close()

    with open("testdata.txt", "r") as trainData, open("testlabels.txt", "r") as trainLabels:
        labelLines = trainLabels.readlines()
        result = []
        correct = 0

        for i, line in enumerate(trainData):
            result.append(applyMultinomial(line, vocabProb1, vocabProb0, wordCount1, wordCount0, probC, totalVocab))

        for i in range(0, len(result) - 1):
            if result[i] == int(labelLines[i].strip()):
                correct += 1

        result = correct / len(result)
        f = open('results.txt', 'a')
        f.write("\n\n")
        f.write("Training Files: traindata.txt, trainlabels.txt\n")
        f.write("Test Files: testdata.txt, testlabels.txt\n")
        f.write("Result: ")
        f.write(str(result))
        f.write("\n")
        f.close()

if __name__ == "__main__":
    main()