import numpy as np

# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # 这个网络节点剩余的处理（转发）能力
        # physical position
        self.cache = None 

# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None

# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None

# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # state
        self.state = EntityState()

class Neighbor(object):
    def __init__(self,link,node):
        self.link = link
        self.node = node

# 网络中的节点
class Node(Entity):
    def __init__(self):
        self.action_callback = None
        # 上一跳的邻居节点及链路，每一个step之后，上一跳中链路
        self.pre_neighbors = None
        # 下一跳邻居节点及链路，应该为Neighbor对象的列表
        # 我们先最好采用数据中心里面分层的架构，规避环路的情况，后面也许会考虑加上环路的情况
        # 这里的len(next_neighbors)就应该等于这个agent的动作的维数（如果egress node数量为1,否则应该还要乘以egress node的数量）
        self.next_neighbors = None

# 继承自node，maddpg中的agent,使用maddpg算法进行决策
class Agent(Node):
    # 网络节点
    def __init__(self):
        super(Agent, self).__init__()
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # 这里将u_range当作分配的概率
        self.u_range = 1.0 
        # state
        self.state = AgentState()
        # action
        self.action = Action()

# 网络中的链路类
class Link(object):  
    # 需要有的信息有：
    # 带宽、流量、链路利用率
    def __init__(self,bandwidth,delay=None):
        self.bandwidth = bandwidth
        self.flow = None
        self.uti = 0
        # 时延可以考虑加上，但是这里的时延还是用比较简单的方法进行实现，也就是经过多少个step，链路上的flow会到下一个节点上
    
    # 计算链路利用率
    def get_uti(self):
        self.uti = self.flow.rate / self.bandwidth
        return self.uti


    def update_flow(self):
        self.flow = flow

# 网络中的流量
class Flow(object):
    # 需要包含的信息有：
    # flow的速率（也就是流量大小），最终的目的地
    def __init__(self,rate,destination):
        self.rate = rate
        self.destination = destination
    # 这个flow是需要分割和合并的，汇聚到同一个节点的最终目的地相同的flow需要合并，然后按照




# multi-agent world
class World(object):
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        # 这里的agents就是网络中除了egress node的节点
        self.agents = []
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2

        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self):
        return self.agents

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    # 这个函数应该放在sceanario中，因为每个sceanario的step的处理逻辑可能有区别
    #def step(self):
    #    pass





    def update_agent_state(self, agent):
        # set communication state (directly for now)
        noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
        agent.state.c = agent.action.c + noise      
