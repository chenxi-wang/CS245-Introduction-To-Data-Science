def pca_reduction(data, num_dims=1):
    '''
    PCA降维的实现
    INPUT:
        data: 原始数据, float (numpy array of N x D)
        num_dims: 降维后的维度数, int
    OUTPUT:
        pca_nd: 降维后的数据, float (numpy array of N x num_dims)
        w: 特征值
        v: 特征向量
    '''
    # 数据中心化
    col_mean = np.mean(data, axis=0)
    centered_data = data - col_mean
    # 求特征值和特征向量
    cov_data = np.cov(centered_data.T)
    w,v = np.linalg.eig(cov_data)
    # 求降维后的数据
    pca_nd = np.dot(centered_data, v[:,:num_dims])
    
    return pca_nd, w, v