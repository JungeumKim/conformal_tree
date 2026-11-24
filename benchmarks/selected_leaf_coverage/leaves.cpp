// Build with the directory containing the previously validated tree.cpp on the include path.
#include "tree.cpp"
namespace {
int cell_id(std::pair<double,double> cell) {
    // On [0,10] all geometric endpoints are exactly representable binary fractions.
    int width_count=static_cast<int>(std::lround(10./(cell.second-cell.first)));
    return width_count+static_cast<int>(std::lround(cell.first*width_count/10.));
}
}
extern "C" void classify_batch(int batch,int N,int K,const double* xs,const double* raw,
                               const double* fitting,const int* query,int* out) {
    for(int i=0;i<batch;++i) {
        const double* x=xs+i*N;
        const double* s=raw+i*N;
        int q=query[i];
#ifdef ORDINARY_TREE
        auto h=ordinary_fit(x,fitting+i*N,N,q,K,false,false,.05);
#else
        auto h=fit(x,fitting+i*N,N,q,q,K,false,false,.05);
#endif
        auto f=fit(x,fitting+i*N,N,q,-1,K,false,false,.05);
        int below=0,calibration_count=0;
        for(int j=0;j<N;++j) {
            if(j==q || x[j]>h.leaf.second || (h.leaf.first!=0 && x[j]<=h.leaf.first)) continue;
            ++calibration_count;
            below+=s[j]<s[q];
        }
        // ceil(.9*(L+1)) using integers; below<rank also covers the +infinity case.
        int rank=(9*(calibration_count+1)+9)/10;
        out[4*i]=cell_id(h.leaf);
        out[4*i+1]=cell_id(f.leaf);
        out[4*i+2]=below<rank;
        out[4*i+3]=calibration_count;
    }
}
