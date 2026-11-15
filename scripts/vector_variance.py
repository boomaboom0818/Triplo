"""Stable sparse-vector sample variance for read-only engineering diagnostics."""
import math

SCHEMA='sparse-coordinate-welford-v2'


class VectorVariance:
    def __init__(self):
        self.count=0;self.coordinates={};self.widths={}

    def add(self,rows):
        for key,row in rows.items():
            if key in self.widths and self.widths[key]!=len(row):raise ValueError('diagnostic vector width changed')
            if any(not math.isfinite(x*x) for x in row):raise ValueError('nonfinite diagnostic second moment')
        self.count+=1
        for key,row in rows.items():
            self.widths[key]=len(row)
            for j,x in enumerate(row):
                if x==0:continue
                n,mean,m2=self.coordinates.get((key,j),(0,0.,0.))
                n+=1;delta=x-mean;mean+=delta/n;m2+=delta*(x-mean)
                if not math.isfinite(m2):raise ValueError('nonfinite diagnostic centered moment')
                self.coordinates[key,j]=(n,mean,m2)

    def trace(self):
        if self.count<2:raise ValueError('at least two diagnostic vectors required')
        # Combine each nonzero-observation group with its missing/zero group.
        # No subtraction of large raw moments: constant vectors give exact zero.
        terms=[m2+mean*mean*(n/self.count)*(self.count-n) for n,mean,m2 in self.coordinates.values()]
        if any(not math.isfinite(x) for x in terms):raise ValueError('nonfinite diagnostic variance')
        return math.fsum(terms)/(self.count-1)
