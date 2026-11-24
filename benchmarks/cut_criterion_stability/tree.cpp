// One-dimensional best-first trees. Both fits receive exactly the same sorted X.
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <utility>
#include <vector>

namespace {
constexpr int m = 20, max_depth = 8;
struct Stats {
    int count = 0;
    long double sum = 0, squares = 0;
    double low = INFINITY, high = -INFINITY;
    void add(double v) {
        ++count; sum += v; squares += static_cast<long double>(v)*v;
        low = std::min(low,v); high = std::max(high,v);
    }
    long double sse() const { return std::max(0.L, squares-sum*sum/count); }
    double range() const { return high-low; }
};
struct Node {
    int l,r,depth;
    double lo,hi;
};
struct Candidate {
    long double gain = -1;
    int boundary = -1;
    double cut = 0;
};
Candidate best(const double* x,const double* s,int hidden, Node v,bool all,bool squared,double rate) {
    Candidate win;
    if (v.depth>=max_depth || v.r-v.l<2*m) return win;
    const double midpoint=(v.lo+v.hi)*0.5;
    int dyadic=std::upper_bound(x+v.l,x+v.r,midpoint)-x;
    if (!all && (dyadic-v.l<m || v.r-dyadic<m)) return win;
    std::vector<Stats> suffix(v.r-v.l+1);
    for (int i=v.r-1;i>=v.l;--i) {
        suffix[i-v.l]=suffix[i-v.l+1];
        if (i!=hidden) suffix[i-v.l].add(s[i]);
    }
    const Stats& parent=suffix.front();
    const long double parent_loss=squared?parent.sse():parent.range();
    Stats left;
    for (int b=v.l+1;b<=v.r-m;++b) {
        if (b-1!=hidden) left.add(s[b-1]);
        if (b-v.l<m || (!all && b!=dyadic) || x[b-1]==x[b]) continue;
        const Stats& right=suffix[b-v.l];
        const long double child_loss=squared ? left.sse()+right.sse() :
                                      0.5*(left.range()+right.range());
        // Same parent/child improvement multiplier as the package, for both losses.
        if (!(parent_loss>(1+rate)*child_loss)) continue;
        const long double gain=parent_loss-child_loss;
        if (gain>win.gain) win={gain,b,all?0.5*(x[b-1]+x[b]):midpoint};
    }
    return win;
}
struct Tree { std::pair<double,double> leaf; std::set<double> cuts; };
Tree fit(const double* x,const double* s,int N,int query,int hidden,int K,bool all,bool squared,double rate) {
    std::vector<Node> leaves{{0,N,0,0.,10.}};
    std::vector<Candidate> candidates{best(x,s,hidden,leaves[0],all,squared,rate)};
    Tree tree;
    while (int(leaves.size())<K) {
        int w=0;
        for (int j=1;j<int(leaves.size());++j)
            if (candidates[j].gain>candidates[w].gain) w=j;
        Candidate c=candidates[w];
        if (c.boundary<0) break;
        Node v=leaves[w], left{v.l,c.boundary,v.depth+1,v.lo,c.cut},
             right{c.boundary,v.r,v.depth+1,c.cut,v.hi};
        tree.cuts.insert(c.cut);
        leaves[w]=left;
        leaves.insert(leaves.begin()+w+1,right);
        if (int(leaves.size())==K) break;
        candidates[w]=best(x,s,hidden,left,all,squared,rate);
        candidates.insert(candidates.begin()+w+1,best(x,s,hidden,right,all,squared,rate));
    }
    for (auto v:leaves) if (v.l<=query && query<v.r) tree.leaf={v.lo,v.hi};
    return tree;
}
}
// Output per method: query mismatch, partition mismatch, leaf count (hidden),
// lower/upper query boundary (full, hidden), leaf count (full).
extern "C" void compare(const double* x,const double* s,int N,int q,int K,double rate,double* out) {
    for (int method=0;method<4;++method) {
        auto f=fit(x,s,N,q,-1,K,method>=2,method%2,rate);
        auto h=fit(x,s,N,q,q,K,method>=2,method%2,rate);
        double* a=out+8*method;
        a[0]=f.leaf!=h.leaf; a[1]=f.cuts!=h.cuts; a[2]=1+h.cuts.size();
        a[3]=f.leaf.first; a[4]=f.leaf.second; a[5]=h.leaf.first; a[6]=h.leaf.second;
        a[7]=1+f.cuts.size();
    }
}
