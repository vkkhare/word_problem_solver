import spacy
from pycorenlp import StanfordCoreNLP
import sys
from string import digits
from pprint import pprint
from itertools import product
from collections import Counter
from nltk.stem.snowball import SnowballStemmer
from spacy import displacy
from spacy.symbols import NUM
nlp = spacy.load("en")
import nltk
from nltk.tree import Tree
""" A Python Class
A simple Python graph class, demonstrating the essential 
facts and functionalities of graphs.
"""
import re
values={}
indexed_list=[]
indexing={}
Small = {
    'zero': 0,
    'one': 1,
    'two': 2,
    'three': 3,
    'four': 4,
    'five': 5,
    'six': 6,
    'seven': 7,
    'eight': 8,
    'nine': 9,
    'ten': 10,
    'eleven': 11,
    'twelve': 12,
    'thirteen': 13,
    'fourteen': 14,
    'fifteen': 15,
    'sixteen': 16,
    'seventeen': 17,
    'eighteen': 18,
    'nineteen': 19,
    'twenty': 20,
    'thirty': 30,
    'forty': 40,
    'fifty': 50,
    'sixty': 60,
    'seventy': 70,
    'eighty': 80,
    'ninety': 90
}

Magnitude = {
    'thousand':     1000,
    'million':      1000000,
    'billion':      1000000000,
    'trillion':     1000000000000,
    'quadrillion':  1000000000000000,
    'quintillion':  1000000000000000000,
    'sextillion':   1000000000000000000000,
    'septillion':   1000000000000000000000000,
    'octillion':    1000000000000000000000000000,
    'nonillion':    1000000000000000000000000000000,
    'decillion':    1000000000000000000000000000000000,
}

class NumberException(Exception):
    def __init__(self, msg):
        Exception.__init__(self, msg)

def text2num(s):
    a = re.split(r"[\s-]+", s)
    n = 0
    g = 0
    for w in a:
        x = Small.get(w, None)
        if x is not None:
            g += x
        elif w == "hundred" and g != 0:
            g *= 100
        else:
            x = Magnitude.get(w, None)
            if x is not None:
                n += g * x
                g = 0
            else:
                raise NumberException("Unknown number: "+w)
    return n + g
mark={}
class Graph(object):
        
        def graph_dict(self):
                return self.__graph_dict
        def __init__(self, graph_dict=None):
                """ initializes a graph object 
                        If no dictionary or None is given, 
                        an empty dictionary will be used
                """
                if graph_dict == None:
                        graph_dict = {}
                self.__graph_dict = graph_dict
                self.__count = {}
        def vertices(self):
                """ returns the vertices of a graph """
                return list(self.__graph_dict.keys())

        def add_vertex(self, vertex):
                """ If the vertex "vertex" is not in 
                self.__graph_dict, a key "vertex" with an empty
                list as a value is added to the dictionary. 
                Otherwise nothing has to be done. 
                """
                if self.get_vertex_key(vertex) not in self.__graph_dict:
                        self.__graph_dict[self.get_vertex_key(vertex)] = []
        def add_edge(self, edge):
                """ assumes that edge is of type set, tuple or list; 
                between two vertices can be multiple edges! 
                """
                if(edge[0] != edge[1]):
                        (vertex1, vertex2) = tuple(edge)              
                        if self.get_vertex_key(vertex1) in self.__graph_dict and self.get_vertex_key(vertex2) in self.__graph_dict:
                                if self.get_vertex_key(vertex2) not in [self.get_vertex_key(w) for w in self.__graph_dict[self.get_vertex_key(vertex1)]]:
                                        self.__graph_dict[self.get_vertex_key(vertex1)].append(vertex2)
                                if self.get_vertex_key(vertex1) not in [self.get_vertex_key(w) for w in self.__graph_dict[self.get_vertex_key(vertex2)]]:
                                        self.__graph_dict[self.get_vertex_key(vertex2)].append(vertex1)
                        elif self.get_vertex_key(vertex1) in self.__graph_dict:
                                self.__graph_dict[self.get_vertex_key(vertex2)] = [vertex1]
                                if self.get_vertex_key(vertex2) not in [self.get_vertex_key(w) for w in self.__graph_dict[self.get_vertex_key(vertex1)]]:
                                        self.__graph_dict[self.get_vertex_key(vertex1)].append(vertex2)
                        elif self.get_vertex_key(vertex2)  in self.__graph_dict:
                                self.__graph_dict[self.get_vertex_key(vertex1)] = [vertex2]
                                if self.get_vertex_key(vertex1) not in [self.get_vertex_key(w) for w in self.__graph_dict[self.get_vertex_key(vertex2)]]:
                                        self.__graph_dict[self.get_vertex_key(vertex2)].append(vertex1)
                        else:
                                self.__graph_dict[self.get_vertex_key(vertex1)] = [vertex2]
                                self.__graph_dict[self.get_vertex_key(vertex2)] = [vertex1]
        def __str__(self):
                res = "vertices: "
                for k in self.__graph_dict:
                        res += str(k) + " "
                res += "\nedges: "
                for edge in self.__generate_edges():
                        res += str(edge) + " "
                return res
        def allot_vertex_key(self, text, sentNum, index):
                stemmed = stemmer.stem(text)
                a = dict() 
                if self.__count.has_key(stemmed):
                        self.__count[stemmed] = self.__count[stemmed]+1
                        a = {"key" : stemmed+str(self.__count[stemmed]),
                             "vertex" : text, "position": [index,sentNum]} 
                else:
                        self.__count[stemmed]=1
                        a = {"key" : stemmed, "vertex" : text, "position" :[index,sentNum]}         
                return a   
        def get_vertex_key(self, vertex):
                return vertex["key"]   
                
def eval_time(g, key):
        """FLAG = {0 (evaluate time),1 (evaluate distance),2 (evaluate speed)}
        """
        dt = g.graph_dict()
        if key in dt:
                distances=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("distance") and mark[w["key"]] == 0]
                speeds=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("speed") and mark[w["key"]] == 0] 
                if (not distances) or (not speeds):
                    distances = [w for w in dt.keys() if stemmer.stem("distance") in w and mark[w] == 0]
                    speeds = [w for w in dt.keys() if stemmer.stem("speed") in w and mark[w] == 0]
                    if len(distances) == 1 and len(speeds) == 1:
                        if distances[0] in values and speeds[0] in values:
                            print "defaulting to time=",values[times[0]],"distance=",values[distances[0]]
                            values[key] = values[distances[0]]/values[speeds[0]]
                            print key,"=",distances[0],"/",speeds[0]
                            return
                for d,s in product(distances,speeds):
                        if (s not in [w["key"] for w in dt[d]]) or (d not in [w["key"] for w in dt[s]]):
                                continue
                        mark[d] = 1
                        mark[s] = 1
                        if d not in values:
                                eval_dist(g,d)        
                        if s not in values:
                                eval_speed(g,s)
                        mark[d] = 0
                        mark[s] = 0        
                        if d in values and s in values:
                                values[key] = values[d]/values[s]  
                                print key,"=",d,"/",s
                                return              
                
def eval_dist(g, key):
        """FLAG = {0 (evaluate time),1 (evaluate distance),2 (evaluate speed)}
        """
        dt = g.graph_dict()
        if key in dt:
                times=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("time") and mark[w["key"]] == 0]
                speeds=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("speed") and mark[w["key"]] == 0] 
                if (not times) or (not speeds):
                    times = [w for w in dt.keys() if stemmer.stem("time") in w  and mark[w] == 0]
                    speeds = [w for w in dt.keys() if stemmer.stem("speed") in w and mark[w] == 0]
                    if len(times) == 1 and len(speeds) == 1:
                        if times[0] in values and speeds[0] in values:
                            print "defaulting to time=",values[times[0]],"speed=",values[speeds[0]]
                            values[key] = values[times[0]]*values[speeds[0]]
                            print key,"=",speeds[0],"*",times[0]
                            return
                for d,s in product(times,speeds): 
                        if (s not in [w["key"] for w in dt[d]]) or (d not in [w["key"] for w in dt[s]]):
                                continue
                        mark[d] = 1
                        mark[s] = 1
                        if d not in values:
                                eval_time(g,d)        
                        if s not in values:
                                eval_speed(g,s)
                        mark[d] = 0
                        mark[s] = 0        
                        if d in values and s in values:
                                values[key] = values[d]*values[s]
                                print key,"=",s,"*",d
                                return
def eval_speed(g, key):
        """FLAG = {0 (evaluate time),1 (evaluate distance),2 (evaluate speed)}
        """
        dt = g.graph_dict()
        if key in dt:
                distances=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("distance") and mark[w["key"]] == 0]
                times=[w["key"] for w in dt[key] if stemmer.stem(w["vertex"]) == stemmer.stem("time") and mark[w["key"]] == 0] 
                if (not distances) or (not times):
                    distances = [w for w in dt.keys() if stemmer.stem("distance") in w and mark[w] == 0]
                    times = [w for w in dt.keys() if stemmer.stem("time") in w and mark[w] == 0]
                    if len(times) == 1 and len(distances) == 1:
                        if times[0] in values and distances[0] in values:
                            print "defaulting to time=",values[times[0]],"distance=",values[distances[0]]
                            values[key] = values[distances[0]]/values[times[0]]
                            print key,"=",distances[0],"/",times[0]
                            return
                for d,s in product(distances,times):
                        if (s not in [w["key"] for w in dt[d]]) or (d not in [w["key"] for w in dt[s]]):
                                continue      
                        mark[d] = 1
                        mark[s] = 1
                        if d not in values:
                                eval_dist(g,d)        
                        if s not in values:
                                eval_time(g,s)
                        mark[d] = 0
                        mark[s] = 0        
                        if d in values and s in values:
                                values[key] = values[d]/values[s]  
                                print key,"=",d,"/",s
                                return
def referenced(w,out,sentNum,index):
        mentions = []
        for w1 in out['corefs']:
                for smallW in out['corefs'][w1]:
                    if (sentNum == smallW['sentNum']) and (smallW['headIndex'] == index) and (not smallW['isRepresentativeMention']):
                            mentions.append(out['corefs'][w1][0])
        return mentions

                        
if __name__ == "__main__":
        stemmer = SnowballStemmer("english")
        file1 = sys.argv[1]
        with open(file1, 'r') as in_file:
                text = in_file.read()
                sents2 = nltk.sent_tokenize(text)
        nlp2 = StanfordCoreNLP("http://localhost:9000")
        props={'annotators':
               'tokenize,dcoref,ssplit,pos,depparse,parse','outputFormat':'json'}
        output = nlp2.annotate(text,props)
        doc = nlp(unicode(open(file1).read().decode('utf8')))        
#        displacy.serve(doc, style='dep')
#        sents2 = sents2[3:]
        sents = [nlp(unicode(sent)) for sent in sents2]
        graph = Graph({})
        s_local_non_dup=[]
        s_list={}
        pprint(output['corefs'])
        count=0
        for i,snt in enumerate(sents):
            indexed_list.append(i)
            indexed_list[i] = [-1]
            for j,w in enumerate(snt):
                indexing.update({count:[i,j+1,w]})
                indexed_list[i].append(count)
                count+=1
        for i in range(0,len(sents)): 
                print(output['sentences'][i]['parse'])
                s_local=[]
                s_local_non_dup={}
                ncl = [l for k in (sents[i]).noun_chunks for l in k]
                j=1
                for w1 in (sents[i]):
                            if ((w1 in ncl or w1.tag_ in ["PRP","PRP$","CD","NN"])) and w1.text not in ['hr','hours','hour','kmph','km','kms','m/s','minutes']:  
                                if w1.tag_ == "CD":        
                                    if ("VB" in w1.head.head.tag_ and w1.head.head.text != "is") or w1.head.head.text == "in" :
                                        v=[]
                                        if w1.head.text in ['hr','hours','hour','minutes']:
                                            v = graph.allot_vertex_key("time",i,j)
                                        elif w1.head.text in ['km','kms']:
                                            v = graph.allot_vertex_key("distance",i,j)
                                        elif w1.head.text in ['kmph']:
                                            v = graph.allot_vertex_key("speed",i,j)
                                        if not v:
                                            continue
                                        graph.add_vertex(v)
                                        s_local.append(v)
                                        s_local_non_dup.update({indexed_list[i][j] : v})
                                        mark[v["key"]]=0
                                    elif (w1.head.head.text == "is"):
                                        if w1.head.text in ['hours','hour','minutes']:
                                            v = [old for old in reversed(s_local) if stemmer.stem(old["vertex"])=="time"] 
                                        elif w1.head.text in ['kms','km']:
                                            v = [old for old in reversed(s_local) if stemmer.stem(old["vertex"])=="distanc"] 
                                        elif w1.head.text in ['kmph']:
                                            v = [old for old in reversed(s_local) if stemmer.stem(old["vertex"])=="speed"]
                                        v=v[0]
                                    else:    
                                        v = [old for old in s_local if stemmer.stem(w1.head.head.head.text) == stemmer.stem(old["vertex"])]
                                        v =v[0]
                                    if w1.text.isdigit():
                                        values[v["key"]] = float(w1.text) 
                                    else:
                                        values[v["key"]] = float(text2num(w1.text))
                                if w1.tag_ in ['NN','NNP','NNPS','NNS','PRP$','PRP']:
                                        mentions = referenced(w1,output,i+1,j)
                                        if not mentions:
                                                v = graph.allot_vertex_key(w1.text,i,j) 
                                                graph.add_vertex(v)
                                                s_local.append(v)
                                                s_local_non_dup.update({indexed_list[i][j] : v})
                                                mark[v["key"]]=0
                                        else:
                                                v =[[mention["headIndex"],mention['sentNum']] for mention in mentions]          
                                                for index in v:
                                                    if s_list and (indexed_list[index[1]-1][index[0]] in s_list):
                                                        s_local += [s_list[indexed_list[index[1]-1][index[0]]]]                                       
                            j+=1
                for w in s_local:
                        for w2 in s_local:
                                graph.add_edge([w,w2])        
                s_list.update(s_local_non_dup)                
        pprint(graph.graph_dict())
#        eval_time(graph,unicode("time2"))
        x = raw_input("enter id of the field needed to calculate ")
        if("time" in x):
            eval_time(graph,unicode(x)) 
        elif("distanc" in x):
            eval_dist(graph,unicode(x))
        else:
            eval_speed(graph,unicode(x))
        print values      
        
