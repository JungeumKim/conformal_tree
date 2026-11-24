#include <algorithm>
#include <cmath>
#include <map>
#include <memory>
#include <numeric>
#include <utility>
#include <vector>
using namespace std;
struct Ends {
    int n=0,lo=-1,second=-1,prev=-1,hi=-1;
    void add(int i){ if(!n) lo=i; if(n==1) second=i; prev=hi; hi=i; ++n; }
    double range(const double* s,int q) const {
        return s[hi==q?prev:hi]-s[lo==q?second:lo];
    }
};
struct Cut { double gain=-1; int j=-1; };
struct Node {
    vector<int> idx,path;
    vector<double> bounds,mid,raw_sorted;
    vector<Ends> children;
    Ends parent;
    Cut best;
    bool ready=false;
    map<int,unique_ptr<Node>> sub;
};
struct Engine {
    const double *x,*s,*raw;
    int N,d,m,K,depth;
    vector<char> marked;
    Node root;
    bool right(int i,int j,double mid) const {return x[i*d+j]>mid;}
    void prepare(Node& v,bool mark){
        if(v.ready) return;
        v.ready=true;
        if((int)v.path.size()>=depth || (int)v.idx.size()<2*m) return;
        v.children.resize(2*d); v.mid.resize(d);
        for(int j=0;j<d;j++) v.mid[j]=.5*(v.bounds[2*j]+v.bounds[2*j+1]);
        for(int i:v.idx){
            v.parent.add(i);
            for(int j=0;j<d;j++) v.children[2*j+right(i,j,v.mid[j])].add(i);
        }
        v.best=best(v,-1,false);
        if(mark) for(int j=0;j<d;j++){
            auto &l=v.children[2*j],&r=v.children[2*j+1];
            if(l.n<m || r.n<m) continue;
            for(const auto* c:{&l,&r}) marked[c->lo]=marked[c->hi]=1;
            if(l.n==m || r.n==m) for(int i:v.idx)
                if(v.children[2*j+right(i,j,v.mid[j])].n==m) marked[i]=1;
        }
    }
    Cut best(Node& v,int q,bool ordinary){
        Cut win;
        if(v.children.empty()) return win;
        double pr=v.parent.range(s,q);
        for(int j=0;j<d;j++){
            auto &l=v.children[2*j],&r=v.children[2*j+1];
            int side=q<0?-1:int(right(q,j,v.mid[j]));
            if(l.n-(ordinary && side==0)<m || r.n-(ordinary && side==1)<m) continue;
            double cr=(l.range(s,q)+r.range(s,q))*.5;
            if(cr==0?pr<=0:pr/cr-1<=.05) continue;
            if(pr-cr>win.gain) win={pr-cr,j};
        }
        return win;
    }
    Node* child(Node* v,int j,int side){
        int code=2*j+side;
        auto &c=v->sub[code];
        if(!c){
            c=make_unique<Node>(); c->bounds=v->bounds; c->bounds[2*j+1-side]=v->mid[j];
            c->path=v->path; c->path.push_back(code);
            for(int i:v->idx) if(right(i,j,v->mid[j])==side) c->idx.push_back(i);
        }
        return c.get();
    }
    struct Leaf { Node* v; bool contains; Cut cut; };
    vector<Leaf> fit(int q,bool ordinary){
        auto leaf=[&](Node* v,bool contains){
            prepare(*v,q==-1);
            return Leaf{v,contains,contains && q>=0?best(*v,q,ordinary):v->best};
        };
        vector<Leaf> active{leaf(&root,true)};
        while((int)active.size()<K){
            int w=-1;
            for(unsigned a=0;a<active.size();a++){
                auto &v=active[a];
                if(v.cut.j>=0 && (w<0 || v.cut.gain>active[w].cut.gain ||
                    (v.cut.gain==active[w].cut.gain && v.v->path<active[w].v->path))) w=a;
            }
            if(w<0) break;
            auto v=active[w]; int j=v.cut.j;
            auto add=[&](int side){
                Node* c=child(v.v,j,side);
                bool contains=v.contains && (q<0 || right(q,j,v.v->mid[j])==side);
                return (int)active.size()+1==K?Leaf{c,contains,{}}:leaf(c,contains);
            };
            auto l=add(0),r=add(1);active[w]=l; active.push_back(r);
        }
        return active;
    }
    void record(Node& v,int q,int* dest){
        if(v.raw_sorted.empty()){
            for(int i:v.idx) v.raw_sorted.push_back(raw[i]);
            sort(v.raw_sorted.begin(),v.raw_sorted.end());
        }
        int below=lower_bound(v.raw_sorted.begin(),v.raw_sorted.end(),raw[q])-v.raw_sorted.begin();
        dest[0]=below<(9*(int)v.idx.size()+9)/10;
        for(unsigned a=0;a<v.path.size();a++) dest[a+1]=v.path[a];
    }
    void run(const double* bounds,int queries,int* out){
        marked.assign(N,0); root.bounds.assign(bounds,bounds+2*d);root.idx.resize(N);
        iota(root.idx.begin(),root.idx.end(),0);
        stable_sort(root.idx.begin(),root.idx.end(),[&](int i,int j){return s[i]<s[j];});
        vector<Node*> by_index(N); for(auto a:fit(-1,false)) for(int i:a.v->idx) by_index[i]=a.v;
        for(int q=0;q<queries;q++) for(int k=0;k<3;k++){
            Node* v=by_index[q];
            if(k>0 && marked[q]) for(auto a:fit(q,k==2)) if(a.contains){v=a.v;break;}
            record(*v,q,out+(q*3+k)*(depth+1));
        }
    }
};
extern "C" void simulate(const double* x,const double* s,const double* raw,const double* bounds,
                         int N,int d,int m,int K,int depth,int queries,int* out){
    Engine e{x,s,raw,N,d,m,K,depth}; e.run(bounds,queries,out);
}
