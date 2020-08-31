# 两个agent的拓扑
import numpy as np
from multiagent.core import World, Agent,Link,Node,Neighbor,Flow
from multiagent.scenario import BaseScenario

class Scenario(BaseScenario):
    def make_world(self):
        world = World()
        # add agents

        node_a = Agent()
        node_b = Node()
        node_c = Node()
        node_d = Node()
        node_e = Node()

        world.agents = [node_a]
        for i, agent in enumerate(world.agents):
            agent.name = 'agent %d' % i

        # 创建链路,本拓扑非常简单，A节点为ingress节点，B,C,D为中继节点，E为egress节点,路线就在BCD中间选择
        link_ab = Link(50)
        link_ac = Link(50)
        link_ad = Link(50)
        link_be = Link(50)
        link_ce = Link(50)
        link_de = Link(50)
        # 连接链路
        node_a.next_neighbors = [Neighbor(link_ab,node_b),Neighbor(link_ac,node_c),Neighbor(link_ad,node_d)]
        node_b.pre_neighbors = [Neighbor(link_ab,node_a)]
        node_b.next_neighbors = [Neighbor(link_be,node_e)]
        node_c.pre_neighbors = [Neighbor(link_ac,node_a)]
        node_c.next_neighbors = [Neighbor(link_ce,node_e)]
        node_d.pre_neighbors = [Neighbor(link_ad,node_a)]
        node_d.next_neighbors = [Neighbor(link_de,node_e)]
        node_e.pre_neighbors = [Neighbor(link_be,node_b),Neighbor(link_ce,node_c),Neighbor(link_de,node_d)]
        # 这里灌入ingress流量
        link_ingress = Link(50)
        link_ingress.flow = Flow(rate=20,destination=node_e)
        node_a.pre_neighbors = [Neighbor(link_ingress,node_a)]
        # make initial conditions

        # 将scenario中的step赋给world(这个地方不太确定，看看有没有bug吧)
        world.step = self.step

        # 在reset_world中也要注意link_ingress中的flow要存在
        self.reset_world(world)
        return world

    # 这里需要将各条链路重新初始化，带宽可以保持不变也可以设置为一个随机值
    def reset_world(self, world):
        # ingress流量->0
        link_ingress = Link(50)
        link_ingress.flow = Flow(rate=None,destination=node_e)
        node_a.pre_neighbors = [Neighbor(link_ingress,node_a)]
        return world
    
    
    # 计算最大链路利用率
    def reward(self, agent, world):
        uti = [1.get_uti() for 1 in self.link]
        max_uti = man(uti)
        return max_uti

    # 观察信息应该包括本节点的剩余处理能力、之前10步的相邻链路利用率、当前的相邻链路利用率等等
    # 最后应该返回ndarray
    def observation(self, agent, world):
        pass

    # 整个世界向前执行一步，具体包括，链路
    # 为了简单起见，我们就不考虑链路和节点的处理时延了，就将每一步当作
    def step(self,world):
        pass
