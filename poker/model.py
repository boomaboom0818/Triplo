"""Public-history estimates, independently owned by each agent."""
from collections import defaultdict
class Opponents:
    def __init__(self): self.counts=defaultdict(lambda:[0,0,0,0.0])
    def update(self, history):
        for player,pos,street,action,pot,call in history:
            row=self.counts[player,pos,street]
            row[0]+=1; row[1]+=action=='fold'; row[2]+=isinstance(action,int)
            if isinstance(action,int): row[3]+=action/max(1,pot)
    def fold_rate(self, player, pos, street):
        n,f,_,_=self.counts[player,pos,street]
        return (f+10)/(n+40),n
