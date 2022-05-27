import ast

class Triple():
    """
        Triple is the class that contains functionality for triples
    """
    __slots__ = "h","r","t"

    def __init__(self,h,r,t):
        """
            Initialise a triples
        Args:
            h (int): Head entity
            r (int): Relation entity
            t (int): Tail entity
        """
        self.h=h
        self.r=r
        self.t=t

    def __eq__(self, other):
        """
            Checks for equality between two triples
        Args:
            other (Triple): Triple object to be compared to

        Returns:
            bool: Returns true if triples are equal, false otherwise 
        """
        return (self.h,self.r,self.t) == (other.h,other.r,other.t)

    def __str__(self):
        
        """
        Converts triple into a string

        Returns:
            str: [h,r,t]
        """
        return "["+str(self.h)+","+str(self.r)+","+str(self.t)+"]"


    def __hash__(self):
        return int(self.pi(self.pi(self.h, self.r), self.t))

    def __str__(self):
        return str(self.h) + " "  + str(self.r) + " " + str(self.t)

    # https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    def pi(self, k1, k2):
        return .5*(k1+k2)*(k1+k2+1)+k2


class DataLoader(object):

    def __init__(self,path,type):
        """
        Init function to initialise the data loader

        Args:
            path (str): Path to folder containing dataset
            type (str): Type of split to load. Type can be "train", "test", "valid"

        Returns:

        """
        self.path = path
        self.headEntities = set()
        self.tailEntities = set()
        self.relations = set()
        self.headDict = {}
        self.tailDict = {}
        self.domain = {}
        self.range = {}
        self.domDomCompatible = {}
        self.domRanCompatible = {}
        self.ranDomCompatible = {}
        self.ranRanCompatible = {}
        self.triple_count_by_pred = {}

        self.relationTotal = 0
        relationPath = path + "relation2id.txt"
        with open(relationPath) as fp:
            self.relationTotal = int(fp.readline())

        self.entityTotal = 0
        entityPath = path + "entity2id.txt"
        with open(entityPath,encoding='utf-8') as fp:
            self.entityTotal = int(fp.readline())

        with open(path + "compatible_relations.txt") as fp:
            self.domDomCompatible = ast.literal_eval(fp.readline())
            self.domRanCompatible = ast.literal_eval(fp.readline())
            self.ranDomCompatible = ast.literal_eval(fp.readline())
            self.ranRanCompatible = ast.literal_eval(fp.readline())

        filePath = path + type + "2id.txt"
        self.list = self.importFile(filePath)

        self.relation_anomaly = {}
        with open(path + "relation2anomaly.txt") as fp:
            line = fp.readline()
            while line:
                pair = line.strip().split()
                self.relation_anomaly[int(pair[0])] = float(pair[1])
                line = fp.readline()

    def importFile(self,filePath):
        """
        Function to import file

        Args:
            filePath (str): Path to file that is required to be opened
            
        Returns:
            
        """
        list = []
        with open(filePath) as fp:
            fp.readline()
            line = fp.readline()
            while line:
                triple = line.strip().split()
                h = int(triple[0])
                t = int(triple[1])
                r = int(triple[2])

                self.headEntities.add(h)
                self.tailEntities.add(t)
                self.relations.add(r)

                if r not in self.headDict:
                    self.headDict[r] = {}
                    self.domain[r] = set()
                if r not in self.tailDict:
                    self.tailDict[r] = {}
                    self.range[r] = set()
                if r not in self.triple_count_by_pred:
                    self.triple_count_by_pred[r] = 0

                if t not in self.headDict[r]:
                    self.headDict[r][t] = set()
                if h not in self.tailDict[r]:
                    self.tailDict[r][h] = set()

                self.headDict[r][t].add(h)
                self.tailDict[r][h].add(t)
                self.domain[r].add(h)
                self.range[r].add(t)

                self.triple_count_by_pred[r] += 1

                triple = Triple(h, r, t)
                list.append(triple)
                line = fp.readline()

        return list

    def getTriples(self):
        """
        Returns the list of triples

        Returns:
            list (list): List of all triples
            
        """
        return self.list

    def getHeadEntities(self):
        """
        Returns all the entites that appear in the head

        Returns:
            headEntities (set): Set of all entities that appear in the head
            
        """
        return self.headEntities

    def getTailEntities(self):
        """
        Returns all the entites that appear in the tail

        Returns:
            tailEntities (set): Set of all entities that appear in the tail
            
        """
        return self.tailEntities

    def getHeadDict(self):
        """
        Returns a dictionary where keys are relations and values are head entities for that relation

        Returns:
            headDict (dict): Head dictionary
            
        """
        return self.headDict

    def getTailDict(self):
        """
        Returns a dictionary where keys are relations and values are tail entities for that relation

        Returns:
            tailDict (dict): Tail dictionary
            
        """
        return self.tailDict

    def getDomain(self):
        """
        Returns a dictionary where keys are relations and values are the domain for that relation

        Returns:
            domain (dict): Domain dictionary
            
        """
        return self.domain

    def getRange(self):
        """
        Returns a dictionary where keys are relations and values are the range for that relation

        Returns:
            range (dict): Range dictionary
            
        """
        return self.range