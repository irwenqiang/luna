double inner_prod(const dVector& feature, const dVector& weight) {
    // size = size
    double prod = .0;
    for(size_t i = 0; i < feature.size(); i++) {
        prod += feature[i] * weight[i];
    }

    return prod;
}
