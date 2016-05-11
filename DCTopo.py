
# coding: utf-8

# In[20]:
# data center topology

#get_ipython().magic(u'matplotlib inline')
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
import random
class FatTree(object):
    def __init__(self,k):
        self.k = k
        self.portNum = k
        self.topo = nx.Graph()
        self.num_core_switch = k**2/4
        self.num_aggregation_switch = k**2/2
        self.num_edge_switch = k**2/2
        self.num_of_switches = self.num_core_switch + self.num_aggregation_switch + self.num_edge_switch
        self.num_of_nodes = self.num_of_switches
        self.switches_ids = [i for i in range(self.num_of_switches)]
        self.core_switches = [i for i in range(0,k**2/4)]
        self.aggregation_switches = [i for i in range(k**2/4,3*(k**2)/4)]
        self.edge_switches = [i for i in range(3*(k**2)/4,5*(k**2)/4)]
        self.generateTopo()
        self.VLANs = [nx.Graph() for i in range(self.k)]
        self.generateVLANs()
    def generateTopo(self):
        # connect aggregation switches with core switchees
        k = self.k
        for i in range(self.k):
            for sw in range(k**2/4+k/2*i,k**2/4+k/2*(i+1)):
                sw_in_pod = sw%(k/2)
                for core_switch in range(k/2*sw_in_pod,k/2*(sw_in_pod+1)):
                    self.topo.add_edge(sw,core_switch)
        # connect aggregation switches with edge switches
        for i in range(self.k):
            for agg_switch in range(k**2/4+k/2*i,k**2/4+k/2*(i+1)):
                for edge_switch in range(3*k*k/4+k/2*i,3*k*k/4+(k/2)*(i+1)):
                    self.topo.add_edge(agg_switch,edge_switch)
    def generateVLANs(self):
        # 共有k个VLAN
        # i是核心层交换机的编号
        k = self.k
        for i in range(k):
            # 先找到和i相连的汇聚层交换机的边
            edges1 = self.topo.edges(i)
            #print edges1
            # 汇聚层交换机
            self.VLANs[i].add_edges_from(edges1)
            aggs = sorted(self.topo.neighbors(i))
            for pod in range(k):
                agg_sw = aggs[pod] # 每个汇聚层交换机和它所在的edge层交换机相连
                for edge_switch in range(3*k*k/4+k/2*pod,3*k*k/4+(k/2)*(pod+1)):
                    self.VLANs[i].add_edge(agg_sw,edge_switch)


    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos,nodesize=100)
        nx.draw_networkx_nodes(self.topo,pos,self.core_switches,node_color='w')
        nx.draw_networkx_nodes(self.topo,pos,self.aggregation_switches,node_color='y')
        plt.title(r"Fat tree $k = {}$".format(self.k))
        plt.axis('off')
        plt.show()


class DCell(object):
            

    def __init__(self,n,k):
        self.n = n
        self.k = k
        self.topo = nx.Graph()
        
        self.portNum = n 
        
        self.num_of_servers = self.compute_num_servers(n,k)
        self.num_of_cells = self.compute_num_cells(n,k)
        self.num_of_switches = self.compute_num_switches()
        # switches' ids
        self.switches_ids = self.compute_switches_ids()
        # servers' ids
        self.servers_ids = [i for i in range(self.num_of_servers)]
        self.num_of_nodes = self.num_of_servers + self.num_of_switches
        
        self.generateTopo([],self.topo,n,k)
    # the number of servers of DCell(k) where each switch has n servers
    
    
    def compute_switches_ids(self):
        ret = [self.num_of_servers+i for i in range(self.num_of_switches)]
        #print ret
        return ret
    
    
    def compute_num_servers(self,n,k):
        t0 = n
        if k == 0:
            return t0
        for i in range(k):
            t0 = t0*(t0+1)
        return t0
    
    
    # the number of DCell(k-1) in DCell(k)
    def compute_num_cells(self,n,k):
        if k == 0:
            return 1
        return self.compute_num_servers(n,k-1) + 1
    
    # one siwtch has n servers
    def compute_num_switches(self):
        return self.num_of_servers/self.n
    
    
    def pref_to_id(self,prefix):
        pref = prefix[:]
        #print "pref = ",
        #print pref
        pref.reverse()
        if len(pref) == 0:
            raise Exception("Prefix cannot be empty!")
        if len(pref) == 1:
            return pref[0]
        sum = pref[0]
        for k in range(1,len(pref)):
            sum += pref[k]*self.compute_num_servers(self.n,k-1)

        #print "id = ",
        #print sum
        
        return sum
    
    
    def generateTopo(self,pref,topo,n,k):
        # part 1
        if(k == 0):
            for i in range(n):
                # connect node [pref,i] to its switch
               
                new_pref = pref[:]
                new_pref.append(i)

                id_of_server = self.pref_to_id(new_pref)

                # 若某个server的pref为[2，3]，则他对应的switch的编号是switches_id的第2个
                if len(new_pref) == 1:
                    id_of_switch = self.switches_ids[0]
                else:
                    id_of_switch = self.switches_ids[new_pref[0]]
                #print id_of_switch, id_of_server
                topo.add_edge(id_of_switch,id_of_server)
            return
        # part 2
        #num_cells = self.num_cells
        for i in range(self.num_of_cells):
            new_pref = pref[:]
            new_pref.append(i)
            self.generateTopo(new_pref,topo,n,k-1)
        # part 3
        for i in range(self.compute_num_servers(n,k-1)):
            for j in range(i+1,self.compute_num_cells(n,k)):
                uid_1 = j-1
                uid_2 = i
                n1 = pref[:]
                n1.append(i)
                n1.append(uid_1)
                n2 = pref[:]
                n2.append(j)
                n2.append(uid_2)
                
                sw1 = self.pref_to_id(n1)
                sw2 = self.pref_to_id(n2)
                topo.add_edge(sw1,sw2)

    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos,nodesize=100)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        
        plt.title(r"DCell $k = {},n = {}$".format(self.k,self.n))
        plt.axis('off')
        plt.show()

class FlattenedButterfly(object):
    def __init__(self,k,n):
        self.k = k
        self.n = n
        self.num_of_switches = int(math.pow(k,n-1))
        self.num_of_servers = int(math.pow(k,n))*2
        self.num_of_nodes = self.num_of_switches + self.num_of_servers
        self.server_portNum = 1
        self.servers_ids = [i for i in range(self.num_of_servers)]
        #print self.servers_ids
        self.switches_ids = [self.num_of_servers+i for i in range(self.num_of_switches)]
        
        self.topo = nx.Graph()
        self.generateTopo(self.topo)
    def generateTopo(self,topo):
        k = self.k
        n = self.n
        # the i-th switch (i is not the id of switch) connects with servers: [2*k,2*k+2*k-1]
        for i in range(self.num_of_switches):
            servers = [s for s in range(2*self.k*i,2*self.k*i+2*self.k)]
            #print servers
            # connect servers and switch
            for s in servers:
                topo.add_edge(s,self.switches_ids[i])
        # connect switches
        for i in range(self.num_of_switches):
            for d in range(1,n):
                for m in range(0,k):
                    j = self.compute_j(i,m,k,d)
                    if i != j and j < len(self.switches_ids) and j >= 0:
                        topo.add_edge(self.switches_ids[i],self.switches_ids[j])
                        
        
    def compute_j(self,i,m,k,d):
        temp = int(math.pow(k,d-1))
        j = i + temp*(m - (int(math.floor(i*1.0/temp))))
        return j
        
    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        
        plt.title(r"Flattened Butterfly $k = {}, n = {}$".format(self.k,self.n))
        plt.axis('off')
        plt.show()


# % matplotlib inline
# import networkx as nx
# import matplotlib.pyplot as plt
import itertools
# import math
class HyperX(object):
    
    def __init__(self,L,S,T):
        self.L = L
        self.S = S
        self.T = T
        self.topo = nx.Graph()
        self.num_of_switches = int(math.pow(S,L))
        self.num_of_servers = T*self.num_of_switches
        
        self.switches_ids= [i for i in range(self.num_of_switches)]
        
        self.servers_ids = [i+self.num_of_switches for i in range(self.num_of_servers)]
        
        self.switches_prefs = self.generate_prefs()
        
        self.num_of_nodes = self.num_of_servers+self.num_of_switches
        
        
        self.prefs_to_id = self.convert_pref_to_id()
        self.generate_topo()
    def generate_prefs(self):
        
        x = [i for i in range(self.S)]
        ret = list(itertools.product(x,repeat=self.L))
        #print "pref: "
        #print ret
        return ret
    def convert_pref_to_id(self):
        d = {}
        for i in range(len(self.switches_prefs)):
            d[self.switches_prefs[i]] = i
        #print d
        return d
    def generate_topo(self):
        # 省点事吧...life is short
        # switch to switch
        for sw1 in self.switches_prefs:
            for sw2 in self.switches_prefs:
                if sw1 != sw2:
                    if sw1[0] == sw2[0] or sw1[1] == sw2[1]:
                        # connnet each other
                        one  = self.prefs_to_id[sw1]
                        another = self.prefs_to_id[sw2]
                        self.topo.add_edge(one,another)
        # switch to server
        for sw in self.switches_ids:
            servers = [s+self.num_of_switches for s in range(sw*self.T,(sw+1)*self.T)]
            for server in servers:
                self.topo.add_edge(server,sw)
        
    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        #print self.topo.edges()
        plt.title(r"HyperX $L = {}, S = {}, T = {}$".format(self.L,self.S,self.T))
        plt.axis('off')
        plt.show()
                
        
        
# h = HyperX(L=2,S=3,T=4)
# h.show()

class SWRing(object):
    """
    k个点组成首位相接的环；每个点再和其他任意四个点相连
    
    随机连接后，不一定保证每个节点的度都为6。因为，前面的节点连完以后，可能导致后面的节点没法连了。
    随便连接很有可能不成功，要多试几次
    
    """
    def __init__(self,k):
        self.k = k
        self.num_switches = k
        self.num_of_nodes = k
        self.switches_ids = [i for i in range(self.k)]
        self.topo = nx.Graph()
        self.random_links = []
        
        self.generate_topo()
        
        
    def generate_topo(self):
        self.success = True # 如果构建完的拓扑不满足条件，那么重新构建一次
        # 先构建环
        for i in range(self.k):
            self.topo.add_edge(i,(i+1)%(self.k))
        # 然后每个节点随机连接四条线
        # 每个点的度数限制为6
        for node in range(self.k):
            already_connected = self.topo.adj[node].keys() #node已经连接的节点
            #print str(node)+" old connections: ",
            #print already_connected
            while len(already_connected)<6:
                left = [i for i in range(self.k) if i not in already_connected and i != node and self.topo.degree(i)<6]
#                 print node,
#                 print "already connections = ",
#                 print already_connected
#                 print "left = ",
#                 print left
                if len(left) == 0:
                    self.success = False
                    break
                random_one = random.choice(left)

                already_connected.append(random_one)
                self.topo.add_edge(random_one,node)
        
                self.random_links.append((random_one,node))
            
                
    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        nx.draw_networkx_edges(self.topo,pos,self.random_links,edge_color='b')
        plt.title(r"Small World Ring nodes = {}".format(self.k))
        plt.axis('off')
        plt.show()

# k*k lattice with random links
class SW2D(object):
    def __init__(self,k):
        self.k = k
        self.num_of_switches = k**2
        self.num_of_nodes = k**2
        self.switches_ids = [i for i in range(self.k**2)]
        self.topo = nx.Graph()
        self.random_links = []
        self.generate_topo()
        self.success = True
    def generate_topo(self):
        
        # 先画lattice
        # 先连接 k-1行
        k =  self.k
        # i : 每个i连接2个节点
        for i in range(k**2-k):
            # 如果i不是最后一列的节点
            if (i+1)%k != 0:
                self.topo.add_edge(i,i+1) # 和右边的节点连接
                self.topo.add_edge(i,k+i) # 和下面的节点连接
            else:
                self.topo.add_edge(i,k+i)
        # 再连接最后一行
        for i in range(k**2-k,k*k-1):
            self.topo.add_edge(i,i+1)
            
        # 最后每一列（行）的首位连
        for i in range(k):
            self.topo.add_edge(i,k**2-k+i)# 第一行的节点和最后一行的节点分别相连
        rows = map(lambda x: x*k,[i for i in range(k)])# 得到第一列的数
        for i in rows:
            self.topo.add_edge(i,i+k-1) # 每一行的第一个元素和该行的最后一个元素相连
            
        # random links
        
        for node in range(k**2):
            already_connected = self.topo.adj[node].keys()
            while len(already_connected)<6:
                left =[i for i in range(k**2) if i != node and i not in already_connected and self.topo.degree(i)<6]
#                 print node,
#                 print "already connections = ",
#                 print already_connected
#                 print "left = ",
#                 print left
                if len(left) == 0:
                    self.success  =  False
                    break
                random_one = random.choice(left)
                already_connected.append(random_one)
                self.topo.add_edge(random_one,node)
                self.random_links.append((random_one,node))
        
        
        
    def show(self):
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        nx.draw_networkx_edges(self.topo,pos,self.random_links,edge_color='b')
        plt.title(r"Small World 2-D Torus nodes = {}".format(self.k**2))
        plt.axis('off')
        plt.show()


# import networkx as nx
# import matplotlib.pyplot as plt
# import random
# 三层。。。有点复杂
class SW3D(object):
    def __init__(self,k):
        self.k = k
        self.n = k*(k-2) # 每层的个数...这只是图的其中一部分，为了方便计算，取k-2行
        self.topo = nx.Graph()
        self.num_of_switches = self.n*3
        self.switches_ids = [i for i in range(self.n*3)]
        self.num_of_nodes = self.n*3
        self.random_links = []
        self.success = True
        self.generate_topo()
    def generate_topo(self):
        k = self.k
        n = self.n
        
        for i in range(3):
            # 先连横线
            ids = [i for i in range(i*n,(i+1)*n)]
            
            for node in ids:
                # 如果node不是最后一列的节点，则它和右边的节点相连
                if (node+1)%k != 0:
                    
                    self.topo.add_edge(node,node+1)
                # 再连接竖线(最后一行不连，到底了)
                # 该点所在的行数
                row = (node%n)/k
                if row < k-3 and row%2 == 0:
                    if node%2 == 1:
                        self.topo.add_edge(node,node+k)
                if row < k-3 and row%2 == 1:
                    if node%2 == 0:
                        self.topo.add_edge(node,node+k)
            # 不同维度之间的节点互相连
            if i == 0 or i == 1:
                self.topo.add_edge(node,node+n)
            
        # 随机连接一条
        for node in self.switches_ids:
            already_connected = self.topo.adj[node].keys()
            left = [i for i in range(self.num_of_switches) if i != node and i not in already_connected and self.topo.degree(i)<6]
            if len(left) == 0:
                self.success = False
                break
            random_one = random.choice(left)
            already_connected.append(random_one)
            self.random_links.append((node,random_one))
            self.topo.add_edge(random_one,node)
        
            
    def show(self):
        
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        nx.draw_networkx_edges(self.topo,pos,self.random_links,edge_color='b')
        plt.title(r"Small World 3-D Torus nodes = {}".format(self.k*(self.k-2)*3))
        plt.axis('off')
        plt.show()


class F10(object):
    
    class Pod(object):
        def __init__(self,id,k,L=2):
            self.id = id
            self.k = k
            self.L = L
            self.p = k/2
            self.contains = self.generate_contains()# 包含的中间一层的交换机，最下层先不考虑
        def generate_contains(self):
            p = self.p
            L = self.L
            p2L = int(math.pow(p,L))
            ret = []
            for i in range(self.id*p+p2L,self.id*p+p2L+p):
                ret.append(i)
            return ret
    
    
    def __init__(self,k,L=3):
        self.k = k
        self.L = L
        self.p = k/2
        self.num_of_switches = 5*(int)(math.pow(k/2,L)) 
        self.num_of_nodes = self.num_of_switches
        self.num_of_cores = (int)(math.pow(k/2,L))
        self.switches_ids = [i for i in range(self.num_of_switches)]
        self.core_ids = [i for i in range(self.num_of_cores)]
        self.middle_ids = [i+self.num_of_cores for i in range(2*(int)(math.pow(k/2,L)))]
        self.pods = [F10.Pod(i,k,L) for i in range(2*(int)(math.pow(k/2,L-1)))]
        self.topo = nx.Graph()
        self.generate_topo()
    def generate_topo(self):
        for pod in self.pods:
            for i in range(len(pod.contains)):
                sw = pod.contains[i] # 第i个交换机为sw
                #print pod.contains
                # A树
                if pod.id%2 == 0:
                    switches_up = [s for s in range(i*self.p,(i+1)*self.p)]# 和当前的交换机连接的上层交换机
                    for switch in switches_up:
                        self.topo.add_edge(sw,switch)
                # B树
                else:
                    switches_up = [s for s in range(self.num_of_cores) if s%2 == i %2]
                    for switch in switches_up:
                        self.topo.add_edge(sw,switch)
                # 连接最下面一层
                t = (int)(math.pow(self.p,self.L)) # t = pow(p,L)
                # 当前 pod对应的最下面的交换机
                switches_down = [s for s in range(pod.id*self.p+3*t,3*t+(pod.id+1)*self.p)]
                #print switches_down
                for switch in switches_down:
                    self.topo.add_edge(sw,switch)
    def show(self):
        
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.core_ids,node_color='w')
        nx.draw_networkx_nodes(self.topo,pos,self.middle_ids,node_color='y')
        plt.title(r"F10 $k={},L={}$".format(self.k,self.L))
        plt.axis('off')
        plt.show()


class LongHop(object):
    def __init__(self,d,m):
        self.d = d
        self.m = m
        self.num_of_switches = int(math.pow(2,d))
        self.num_of_nodes = self.num_of_switches
        self.switches_ids = [i for i in range(self.num_of_switches)]
        self.topo = nx.Graph()
        self.generate_topo()
    def generate_topo(self):
        # 节点：使用二进制表示
        nodes = list(itertools.product([0,1],repeat=self.d))
#         print "Nodes: ",
#         print nodes
        A = np.eye(self.d,dtype=int)
        B = np.zeros((self.d,(self.m - self.d)),dtype=int)
        
        for r in range(self.d):
            for c in range(self.m-self.d):
                B[r][c] = random.choice([0,1])
        G_temp = np.hstack((A,B))
        G = G_temp.transpose()
#         print "rows in G"
#         for row in G:
#             print row
            
        for x in nodes:
            # 每个节点x和m个二进制数相异或（矩阵正好是m行）
            # 将得到的二进制数，转化为十进制数d。
            # x和d相连
            for row in G:
                h = []
                for c in row:
                    h.append(c)
                node_binary = self.XOR(x,h)
                #print node_binary,
                #print(" 转化为十进制为： ")
                node = self.convert2int(node_binary)
                #print node
                self.topo.add_edge(self.convert2int(x),node)
                
        
    def XOR(self,list_a,list_b):
        ret = []
        assert len(list_a) == len(list_b)
        for i in range(len(list_a)):
            temp = list_a[i]^list_b[i]
            ret.append(temp)
        return ret
    def convert2int(self,li):
        length = len(li)
        s = 0
        for i in range(length):
            s = s*2 + li[i]
        return s
    
    def show(self):
        
        pos = nx.circular_layout(self.topo)
        nx.draw_networkx(self.topo,pos)
        nx.draw_networkx_nodes(self.topo,pos,self.switches_ids,node_color='w')
        plt.title(r"Long Hop $d = {}, m = {}$".format(self.d,self.m))
        plt.axis('off')
        plt.show() 


