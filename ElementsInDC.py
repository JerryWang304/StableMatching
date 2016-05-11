
# coding: utf-8

# In[15]:

# Elements in DC simulation
#from DCTopo import FatTree
import simpy
import random
import networkx as nx

class Flow(object):   
    # def __init__(self,src,des,demand,size):
    # src and des are integer
    def __init__(self,id,time,src,des,size,bandwidth_demand):
        self.begin_time = time
        self.src = src
        self.end_time = time
        self.des = des
        self.id = id
        self.size = size # flow size such as 1Gb
        self.bandwidth_demand = bandwidth_demand # bandwidth demand of each flow
    def __repr__(self):
        return "from {} to {},size: {:.2f}bits,bandwidth_demand:{:.2f}kbps,begin_time: {:.2f},end_time:{:.2f}".format(self.src,self.des,self.size,self.bandwidth_demand,self.begin_time,self.end_time)
        

# class FlowGenerator(object):
#     """
#     generate a flow to receiver
#     the destination of the flow is randomly choosen from left switches
#     1Gbps = 10^6 kbps
#     """
#     def __init__(self,env,god,out,controller,initial_delay=0,end_time=float('inf'),debug=False):
#         self.env = env
#         self.god = god
#         self.controller = controller
#         #self.interval = interval
#         self.initial_delay = initial_delay
#         self.end_time = end_time
#         self.out = out # out is a switch
#         self.flow_generated = 0
#         self.debug = debug
#         self.action = self.env.process(self.run())
#     # 每个流的带宽需求
#     def bandwidth_demand(self):
#         return random.uniform(1,10000) # 1~10000 kbps
#     # 每个flow的大小
#     def flow_size(self):
#         return random.uniform(1024*8,1024*20*8)  # 1~20KB
#     def run(self):
#         pass
# class FlowGenerator_FatTree(FlowGenerator):
#     """
#     generate a flow to receiver
#     the destination of the flow is randomly choosen from left switches
#     1Gbps = 10^6 kbps
#     """

#     def run(self):
#         while True:
#             yield self.env.timeout(self.initial_delay)
#             while self.env.now < self.end_time:
                
#                 #assert self.out is not None
#                 # 同一时刻向out注入flows
#                 flows = []
#                 for receiver in self.out:
#                     destinations = self.god.topo.edge_switches[:].remove(receiver)
#                     des = random.choice(destinations)
#                     flow = Flow(self.flow_generated,self.env.now,receiver.id,des,self.flow_size(),self.bandwidth_demand())
#                     flows.append(flows)
#                     #print flow
#                     if self.debug:
#                         print "生成的流："
#                         print flow
#                     self.flow_generated += 1
#                     self.controller.put(flow)
# 收集flows 
import operator 
import numpy     
import math             
class Controller(object):
    """
    目前只适合Fat tree
    生成流
    调度流
    Stable matching 在这里实现
    """
    def __init__(self,env,topo,out):
        self.env = env
        self.topo = topo # 类
        self.out = out # 交换机的id 把流发送给这些out
        self.link_bandwidth = 10**7 # kbps
        self.random_link_bandwidth = self.generate_link_bandwidth()
        self.flow_generated = 0
        self.flows_in_routing_class = {} # 每个routing class对应的flows
        self.num_of_routing_class = 8#暂时设置8组flows    
        self.store = simpy.Store(env)# 收集到达终点的流
        self.flows = self.generate_flows() # 生成流
        self.capacity =  self.flow_generated/len(self.topo.VLANs)# 每个vlan最多能容纳的个数，都一样
        # if stable_matching:
        #     self.stable_matching_assignment() # 给每个流分配VLAN
        # elif wait_and_hop:
        #     self.wait_and_hop_assignment()

    @property
    def ave_time(self):
        
        durations = [] # 每个流持续的时间
        #print len(self.store.items)
        for flow in self.store.items:
            duration = flow.end_time - flow.begin_time
            durations.append(duration)
        assert len(durations) > 0
        average = sum(durations)*1.0/len(durations)
        return average

    def put(self,el):
        self.flows_store.put(el)
    # 每个流的带宽需求
    def bandwidth_demand(self):
        return random.uniform(10000,200000) # kbps
    # 每个flow的大小
    def flow_size(self):
        return random.uniform(1024*8,1024*20*8)  # 1~20KB (bits)
    
    def generate_flows(self):
        flows = []
        for i in range(self.num_of_routing_class):
            temp = []
            for receiver in self.out:
                destinations = self.out[:]
                destinations.remove(receiver)
                #print destinations
                des = random.choice(destinations)
                flow = Flow(self.flow_generated,self.env.now,receiver,des,self.flow_size(),self.bandwidth_demand())
                flow.routing_class = i
                flows.append(flow)
                temp.append(flow)
                self.flow_generated += 1
            self.flows_in_routing_class[i] = temp
        return flows

    

    #stable machting
    def stable_matching_assignment(self):
        flows = self.flows
        # vlan选flow
        # 每个VLAN的优先选择bandwidth demand小的flow
        perference_list_of_vlans = {}
        flows_temp = sorted(flows,key=operator.attrgetter('bandwidth_demand')) 
        for i in range(len(flows_temp)):
            flows_temp[i].preference = i
        for vlan_id in range(len(self.topo.VLANs)):
            perference_list_of_vlans[vlan_id] = flows_temp # 
        # 测试：每个flow随机生成一个perference list
        perference_list_of_flows = {}
        num_of_vlans = len(self.topo.VLANs)
        num_of_flows = len(flows)
        proposed = numpy.zeros((num_of_flows,num_of_vlans))
        engaged = {i:[] for i in range(num_of_vlans)}

        # flow选vlan
        #print self.compute_max_linkutilization(self.topo.VLANs[0],flows)
        #print self.compute_max_linkutilization(self.topo.VLANs[3],flows)
        # 随机选择一半的half_flows，计算各个VLAN在当前half_flows下的最大链路利用率
        for i in range(self.num_of_routing_class):

            # 当前routing class里的flows
            those_flows = self.flows_in_routing_class[i]
            #print those_flows
            half_flows = those_flows[:]
            left_routing_class = [c for c in range(self.num_of_routing_class)]
            left_routing_class.remove(i)
            #print left_routing_class
            num_of_left_routing_class = self.num_of_routing_class/2-1 # 加自己，总共选择一半
            select_routing_class = random.sample(left_routing_class,num_of_left_routing_class)
            for rc in select_routing_class:
                half_flows += self.flows_in_routing_class[rc]
            #print half_flows
            # 每个vlan对应的最大链路利用率
            vlan_to_max_utilization = []
            # 最大链路利用率对应的VLAN
            max_utilization_to_vlan = {}

            for vlan_id in range(len(self.topo.VLANs)):
                max_utilization = self.compute_max_linkutilization(self.topo.VLANs[vlan_id],half_flows)
                vlan_to_max_utilization.append(max_utilization)
                max_utilization_to_vlan[max_utilization] = vlan_id
            
            vlan_to_max_utilization.sort() # 从小到大排序
            #print vlan_to_max_utilization
            pre_list = []
            for el in vlan_to_max_utilization:
                pre_list.append(max_utilization_to_vlan[el])
            #print pre_list
            for flow in self.flows_in_routing_class[i]:
                perference_list_of_flows[flow.id] = pre_list
            
        free_flows = flows[:]
        
        while len(free_flows) > 0:
            flow = free_flows[0]
            for i in range(len(perference_list_of_flows[flow.id])):
                if proposed[flow.id][i] == 0:
                    vlan = perference_list_of_flows[flow.id][i]
                    vlan_id = i
                    proposed[flow.id][i] = 1
                    break
            if  len(engaged[vlan_id]) < self.capacity:
                engaged[vlan_id].append(flow)
                free_flows.remove(flow)
            else:
                # 如果满了，先把flow插入进去按优先级排序；并且按照当前flow的优先级插入，舍弃优先级最低的flow
                engaged[vlan_id].append(flow)
                engaged[vlan_id].sort(key=operator.attrgetter('preference'))
                last = engaged[vlan_id][len(engaged[vlan_id])-1]

                engaged[vlan_id].remove(last)
                if last.id != flow.id:
                    free_flows.remove(flow)
                    free_flows.append(last)

        for v, fs in engaged.items():
            for f in fs:
                f.vlan = self.topo.VLANs[v]
        # 计算此时的最大链路利用率
        link_to_utilization = {} # 例如(1,2) : 0.23
        for flow in flows:
            #print flow.vlan_id,
            path = nx.shortest_path(flow.vlan,flow.src,flow.des)
            i ,j= 0, 1
            while True:
                first, second = path[i], path[j]
                if first > second:
                    first, second = second, first
                link = (first,second)
                # 把每个流的demand加到link上
                
                if not link in link_to_utilization.keys():
                    link_to_utilization[link] = flow.bandwidth_demand/self.random_link_bandwidth[link]
                else:
                    link_to_utilization[link] += flow.bandwidth_demand/self.random_link_bandwidth[link]

                if j == len(path)-1:
                    break
                i += 1
                j += 1

  
        max_utilization = -1
        for v in link_to_utilization.values():
            if v > max_utilization:
                max_utilization = v
        
        return max_utilization

    # 计算flows在拓扑topo下的最大链路利用率
    def generate_link_bandwidth(self):
        """
        每个link的带宽是不固定的
        """
        link_bandwidth = {}
        for node in range(self.topo.num_of_nodes):
            edges = self.topo.topo.edges(node)
            for edge in edges:
                if edge[1] > edge[0]:
                    link_bandwidth[edge] = random.uniform(self.link_bandwidth/2,self.link_bandwidth)
        return link_bandwidth
    def compute_max_linkutilization(self,topo,flows):
        """
        计算flows在topo下的最大链路利用率
        """
        # 每条flow经过的link
        link_to_utilization = {} # 例如(1,2) : 0.23
        for flow in flows:
            path = nx.shortest_path(topo,flow.src,flow.des)
            i ,j= 0, 1
            while True:
                first, second = path[i], path[j]
                if first > second:
                    first, second = second, first
                link = (first,second)
                # 把每个流的demand加到link上
                
                if not link in link_to_utilization.keys():
                    link_to_utilization[link] = flow.bandwidth_demand/self.random_link_bandwidth[link]
                else:
                    link_to_utilization[link] += flow.bandwidth_demand/self.random_link_bandwidth[link]

                if j == len(path)-1:
                    break
                i += 1
                j += 1
        #print link_to_utilization
        max_utilization = -1
        for v in link_to_utilization.values():
            if v > max_utilization:
                max_utilization = v
        return max_utilization

    def wait_and_hop_assignment(self,beta=1):
        # 先随机分配一个vlan
        alpha = 0.005
        num_of_vlans = len(self.topo.VLANs)
        vlan_assignment = []
        for i in range(self.num_of_routing_class):
            vlan = random.choice([v for v in range(num_of_vlans)])
            vlan_assignment.append(vlan)
        #print vlan_assignment
        # 将每个流所属的VLAN进行更新
        for i in range(self.num_of_routing_class):
            these_flows = self.flows_in_routing_class[i]

            for flow in these_flows:
                flow.vlan_id = vlan_assignment[i]
                flow.vlan = self.topo.VLANs[flow.vlan_id]
        # 按照当前的VLAN分配方法计算Phi(x)：最大链路利用率
        old_phi = 10000000
        stay_time_of_vlan_assignment = {}
        vlan_assignment_to_max_utilization = {}
        while True:
            # phi：最大链路利用率
            # 计算当前划分下的最大链路利用率
            flows = []
            for i in range(self.num_of_routing_class):
                these_flows = self.flows_in_routing_class[i]
                flows += these_flows
            link_to_utilization = {} # 例如(1,2) : 0.23
            for flow in flows:
                #print flow.vlan_id,
                path = nx.shortest_path(flow.vlan,flow.src,flow.des)
                i ,j= 0, 1
                while True:
                    first, second = path[i], path[j]
                    if first > second:
                        first, second = second, first
                    link = (first,second)
                    # 把每个流的demand加到link上
                    
                    if not link in link_to_utilization.keys():
                        link_to_utilization[link] = flow.bandwidth_demand/self.random_link_bandwidth[link]
                    else:
                        link_to_utilization[link] += flow.bandwidth_demand/self.random_link_bandwidth[link]

                    if j == len(path)-1:
                        break
                    i += 1
                    j += 1

      
            max_utilization = -1
            for v in link_to_utilization.values():
                if v > max_utilization:
                    max_utilization = v
            phi = max_utilization
            # 统计一下每个vlan的最大链路利用率
            vlan_assignment_to_max_utilization[str(vlan_assignment)] = max_utilization


            _lambda = alpha*(num_of_vlans-1)*math.exp(beta*phi)
            # 生成num_of_routing_class个随机数
            counting_down_number = []
            for i in range(self.num_of_routing_class):
                timer = random.expovariate(_lambda)
                counting_down_number.append(timer)
            # 找到最小值的随机数和相应的下标（即routing class）
            #print "counting_down_number:",
            #print counting_down_number
            min_timer = 1e9
            stay_routing_class = 0
            for i in range(self.num_of_routing_class):
                if counting_down_number[i] < min_timer:
                    min_timer = counting_down_number[i]
                    stay_routing_class = i
            # 记录当前的停留时间
            if not str(vlan_assignment) in stay_time_of_vlan_assignment.keys():
                stay_time_of_vlan_assignment[str(vlan_assignment)] = 0
            else:
                stay_time_of_vlan_assignment[str(vlan_assignment)] += min_timer
            #print "stay_routing_class:",
            #print stay_routing_class
            # 更新vlan assignment
            # 除了stay_routing_class以外，其它的随机从剩下的VLAN中随机选择一个（不包括自己）
            #print "之前的vlan assignment:",
            #print vlan_assignment
            for i in range(self.num_of_routing_class):
                if i != stay_routing_class:
                    old_vlan = vlan_assignment[i]
                    vlans = [v for v in range(num_of_vlans)]
                    vlans.remove(old_vlan)
                    #  随机选择一个
                    new_vlan = random.choice(vlans)
                    #print old_vlan,new_vlan,
                    #print i
                    vlan_assignment[i] = new_vlan
                    #print vlan_assignment[i]
                    #print old_vlan,new_vlan

            # 更新每个流的vlan
            #print "随机选择vlan：",
            #print vlan_assignment
            for i in range(self.num_of_routing_class):
                these_flows = self.flows_in_routing_class[i]
                for flow in these_flows:
                    flow.vlan_id = vlan_assignment[i]
                    flow.vlan = self.topo.VLANs[flow.vlan_id]
            new_phi = phi
            
            #print "current phi: ",
            #print phi,
            if abs(new_phi-old_phi) < 1e-5:
                break
            old_phi = phi
        # print '\n'
        stay_time_of_vlan_assignment = sorted(stay_time_of_vlan_assignment.items(),key=lambda x: x[1],reverse=True)
        #print "stay_time_of_vlan_assignment: "
        #print stay_time_of_vlan_assignment
        min_max_utilization = 100

        #print "停留时间最长的链路利用率：",
        # 从前三个停留时间最长的vlan_assignment中找到最小的一个
        for i in range(3):
            #print stay_time_of_vlan_assignment[i][0]
            if vlan_assignment_to_max_utilization[str(stay_time_of_vlan_assignment[i][0])] < min_max_utilization:
                min_max_utilization = vlan_assignment_to_max_utilization[str(stay_time_of_vlan_assignment[i][0])]
                string_of_final_vlan_assignment = stay_time_of_vlan_assignment[i][0] # type:str
        #print "min_max_utilization: ",
        #print min_max_utilization
        # print "string_of_final_vlan_assignment: ",
        # print string_of_final_vlan_assignment
        final_vlan_assignment = []
        # 提取所有数字（vlan的个数必须小于10个）
        s = filter(str.isdigit,string_of_final_vlan_assignment)
        for el in range(len(s)):
            final_vlan_assignment.append(int(s[el]))
        #print final_vlan_assignment
        # 根据最后的结果更新流的vlan
        for i in range(self.num_of_routing_class):
            these_flows = self.flows_in_routing_class[i]

            for flow in these_flows:
                flow.vlan_id = final_vlan_assignment[i]
                flow.vlan = self.topo.VLANs[flow.vlan_id]
        return min_max_utilization


    def randomly_vlan_assignment(self):
        """每个flow随机选择一个VLAN
        暂时没有用到routing class
        """
        flows = self.flows

        perference_list_of_vlans = {}
        flows_temp = sorted(flows,key=operator.attrgetter('bandwidth_demand')) # 每个VLAN的优先选择bandwidth demand小的flow
        for i in range(len(flows_temp)):
            flows_temp[i].preference = i
        for vlan_id in range(len(self.topo.VLANs)):
            perference_list_of_vlans[vlan_id] = flows_temp # 
        # 测试：每个flow随机生成一个perference list
        perference_list_of_flows = {}
        num_of_vlans = len(self.topo.VLANs)
        num_of_flows = len(flows)
        proposed = numpy.zeros((num_of_flows,num_of_vlans))
        engaged = {i:[] for i in range(num_of_vlans)}
        for flow in flows:
            perference_list_of_flows[flow.id] = random.sample(self.topo.VLANs,num_of_vlans)
            
        free_flows = flows[:]
            
        while len(free_flows) > 0:
            flow = free_flows[0]
            for i in range(len(perference_list_of_flows[flow.id])):
                if proposed[flow.id][i] == 0:
                    vlan = perference_list_of_flows[flow.id][i]
                    vlan_id = i
                    proposed[flow.id][i] = 1
                    break
            if  len(engaged[vlan_id]) < self.capacity:
                engaged[vlan_id].append(flow)
                free_flows.remove(flow)
            else:
                # 如果满了，先把flow插入进去按优先级排序；并且按照当前flow的优先级插入，舍弃优先级最低的flow
                engaged[vlan_id].append(flow)
                engaged[vlan_id].sort(key=operator.attrgetter('preference'))
                last = engaged[vlan_id][len(engaged[vlan_id])-1]

                engaged[vlan_id].remove(last)
                if last.id != flow.id:
                    free_flows.remove(flow)
                    free_flows.append(last)

        for v, fs in engaged.items():
            for f in fs:
                f.vlan = self.topo.VLANs[v]


class Port(object):
    # port is designed for forwarding flows
    """
    implement ECMP
    src: where the flows originate
    """
    def __init__(self,env,god,src,rate,controller):
        self.env = env
        self.src = src
        self.rate = rate
        self.god = god # it knows all the switches...
        self.topo = god.topo.topo # nx Graph()
        self.out = controller
        self.action = self.env.process(self.run())
    def run(self):
        while True:
            flow = yield self.src.store.get() # switches里面有flows才会转发
            # 

            if flow.des == self.src.id:
                flow.end_time = self.env.now
                self.out.store.put(flow)
                #print flow,
                #print " -> arriving time %.8f" % self.env.now
                continue
            # compute the next hop
            
            paths = nx.all_shortest_paths(self.topo,self.src.id,flow.des)
            next_hops = []
            for path in paths:
                if len(path)>1:
                    next_hops.append(path[1])
            next_hop = random.choice(next_hops)
            # print next_hop,
            # forwarding to next_hop
            target = self.src.id
            for sw in self.god.all_nodes:
                if sw.id == next_hop:
                    target = sw
                    break
            #print target
            if target:
                flow.src = target.id
                yield self.env.timeout(flow.size*1.0/self.rate)
                target.store.put(flow)

class VLANsAssignmentPort(Port):
    def run(self):
        while True:
            flow = yield self.src.store.get()
            #print "frowarding ",
            #print flow
            if flow.des == self.src.id:
                flow.end_time = self.env.now
                self.out.store.put(flow)
                #print flow
                continue
            # compute the next hop
            # 只会得到一条路径
            path = nx.shortest_path(flow.vlan,self.src.id,flow.des)
            #print path
            assert path is not None and len(path) > 1
            next_hop = path[1]
            # forwarding to next_hop (target)
            target = self.src.id
            for sw in self.god.all_nodes:
                if sw.id == next_hop:
                    target = sw
                    break
            #print target
            if target:
                flow.src = target.id
                yield self.env.timeout(flow.size*1.0/(flow.bandwidth_demand*1000))
                target.store.put(flow)


class Switch(object):
    # 10Gbps = 10**7 kbps
    def __init__(self,env,god,id,portNum=1,qlimit=None,rate=10**7,controller=None):
        self.env = env
        self.id = id
        self.god = god
        self.qlimit = qlimit
        self.store = simpy.Store(env)# store flows
        self.portNum = portNum
        self.ports = [VLANsAssignmentPort(env,god,self,rate,controller) for i in range(self.portNum)] 
        self.flow_queue = []

    def __repr__(self):
        return "Switch:{}".format(self.id)
            
class God(object):
    
    def __init__(self,env,DCTopo,rate,controller):
        self.topo = DCTopo 
        self.env = env
        self.rate = rate
        self.controller = controller   
        self.switches = self.generate_switches() # Switch object
        self.all_nodes = self.switches
        self.inject_flows() # 将生成好的流注入到交换机中
        
    def generate_switches(self):
        
        sw = []
        
        for id in self.topo.switches_ids:
            switch = Switch(self.env,self,id,portNum=self.topo.topo.degree(id),qlimit=None,rate=self.rate,controller=self.controller)
            sw.append(switch)
        return sw
        
    
    def inject_flows(self):
        # generate flows first
        # 给每个Edge switch生成流
        
        flows = self.controller.flows
        for flow in flows:
            src_switch = flow.src
            self.switches[src_switch].store.put(flow)




