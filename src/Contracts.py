#Team Members: Zhaoguo Zhu/Piotr Nojeszewski
#Clinet: Dr. Kaija Schilde
#Boston University Spark Engineering Project
#Outsourcing Security

import csv

# Disclaimer: no ML in this code.
class Contracts:
    """ Read contract data and sort categories by occurance """

    def __init__(self, source):
        """ Initialize the model.
            Input
                source [String]
                    path to the !-separated file. The file must have the standard DCADS contract data format.
            Return
                None
        """
        self.source = source


    def __setContracts(self):
        """ Read input data and set contracts
            Input
                None
            Return
                None
        """
        X = []
        try:
            with open(self.source, encoding='windows-1252') as csv_file:
                csv_reader = list(csv.reader(csv_file, delimiter='!'))
                line_count = 0
                for row in csv_reader:
                    X.append([])
                    for i in range(len(row)):
                        X[line_count].append(row[i])
                    line_count += 1
        except:
            raise Exception("Error while reading the file.")
        self.contracts = X

    def __setLabels(self):
        """ Extract labels.
            Input
                None
            Return
                None
        """
        Y = []
        for x in self.contracts:
            try:
                Y.append(x[34])
            except:
                pass

        labels = dict()
        for y in Y:
            if y in labels:
                labels[y] += 1
            else:
                labels[y] = 1

        self.labels = dict()
        for (i,key) in enumerate(labels):
            newKey = key
            for j in range(len(newKey)):
                if newKey[-1] == " ":
                    newKey = newKey[:-1]
                else:
                    break
            self.labels[newKey] = labels[key]

    def __setSortedLabels(self):
        """ Extract labels.
            Input
                None
            Return
                None
        """
        try:
            total = 0
            self.sortedLabels = []
            for (i,key) in enumerate(self.labels):
                total = i
                self.sortedLabels.append([key, self.labels[key]])
            self.sortedLabels.sort(key=(lambda x:x[1]), reverse=True)
        except:
            raise Exception("Error while sorting labels")

    def run(self, verbose=False):
        """ Run contract analysis.
            Input
                (optional) verbose [bool]
                    If set to True, logs will be printed.
            Return
                None
        """
        if(verbose):
            print("Conctracts: Starting run()")
        self.__setContracts()
        if(verbose):
            print("Conctracts: Read input data")
        self.__setLabels()
        if(verbose):
            print("Contracts: Labels created")
        self.__setSortedLabels()
        if(verbose):
            print("Contracts: Labels sorted")
            print("Contracts: Finished run()")

    def getLabels(self):
        """ Get label dictionary
            Input
                None
            Return
                labels [dict]
        """
        try:
            return self.labels
        except:
            raise Exception("First start the model with run()")

    def getSortedLabels(self):
        """ Get sorted label list
            Input
                None
            Return
                sorted labels [list]
        """
        try:
            return self.sortedLabels
        except:
            raise Exception("First start the model with run()")


#X = import_data("web2005.txt")
