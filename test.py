from data_preoperator import *
from layer import *
from run import *
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
def testAcc():
    a = numpy.array([2,2,1,1,3,3])
    b = numpy.array([3,3,2,2,1,1])
    labSize = 3
    print_accuracy(a,b,labSize)

def testCNN():
    module = CNN()
    module.load_state_dict(torch.load("./module/CNN.pkl"))

    batch_size = 100
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        images = F.normalize(images)
        images = images.view(images.shape[0],-1)
        if i == 0:
            encoder,decoder = module(images)
            decoder = decoder.view(decoder.shape[0],1,28,28)
            show_img(decoder,batch_size)
            break

def testGetAMatrix():
    a = torch.FloatTensor([[1,2,3],[2,3,4],[3,4,5]])
    print("input is ")
    print(a)
    print("output is")
    out = get_AMatrix(a, "fc")
    print(out)

def testSelfExpression():
    batch_size = 100
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    cnn = CNN()
    cnn.load_state_dict(torch.load("./module/CNN.pkl"))


    for i, (images, labels) in enumerate(train_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        if i == 0:
            encoder,decoder = cnn(images)
            # show_img(encoder,batch_size)

            Z = images.view(images.size(0), -1)
    
            module = TrainSelfExpressionLayer(Z,batch_size)

            images = images.view(images.size(0),-1)

            C = adjustMatrixC(module.fc[0].weight)

            ZC = C.mm(images)

            ZC = ZC.view(ZC.size(0),1,28,28)
            images = images.view(images.size(0),1,28,28)
            show_img(images, batch_size)
            show_img(ZC, batch_size)
            break

def testKnn():
    a = torch.FloatTensor([[1,2,3],[2,3,4],[4,5,6]])
    print("input X is")
    print(a)
    b = knn_affinity(a,2)
    print("output is")
    print(b)

def testGetKneibor():
    isLabels = True
    if isLabels:
        print("labels")
        labels = torch.FloatTensor([1,1,2,3,2,1])
        print("input is")
        print(labels)
        b = getKneibor(lables=labels)
        print("outputs")
        print(b)

        print("1 and 0 is neibor?")
        print(isNeibor(b,isLabels,1,0))
        print("2 and 1 is neibor?")
        print(isNeibor(b,isLabels,2,1))
        print("2 and 0 is neibor?")
        print(isNeibor(b,isLabels,2,4))
    else:
        print("unlabels")
        a = torch.FloatTensor([[1,2,3],[2,3,4],[4,5,6]])
        print("input X is")
        print(a)
        b = getKneibor(a,2)
        print("output is")
        print(b)
        print("1 and 0 is neibor?")
        print(isNeibor(b,1,0))
        print("2 and 1 is neibor?")
        print(isNeibor(b,2,1))
        print("2 and 0 is neibor?")
        print(isNeibor(b,2,0))

def testSiamese():
    batch_size = 10
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    siamese = SiameseNetwork()
    siamese.load_state_dict(torch.load("./module/Siamese2.pkl"))
    loss_func = ContrastiveLoss()
    images, labels = next(iter(train_loader))
    images = get_variable(images)
    labels = get_variable(labels)

    # kneibor = getKneibor(X=images.view(images.shape[0],-1),n_nbrs=batch_size)
    # print(kneibor)
    neibor = getKneibor(lables=labels)
    input = torch.FloatTensor([])
    for n in range(images.shape[0]):
        input = torch.cat((input,siamese.forward_once(images[n:n+1])),0)

    A = torch.zeros(images.shape[0],images.shape[0])
    for n in range(images.shape[0]):
        for m in range(images.shape[0]):
            A[n][m] = loss_func(input[n:n+1],input[m:m+1],1)
    print(A)
    A = torch.clamp(16 - squared_distance(input), min=0.0)
    print(A)
    show_img(images,batch_size)

def testsquared_distance():
    a = torch.FloatTensor([[1,2,3],[2,3,4],[3,4,5]])
    print(a)
    print("test loss")
    a1 = a[0:1,:]
    a2 = a[2:3,:]
    print("input1 is", a1)
    print("input2 is", a2)
    loss_func = ContrastiveLoss()
    print("ContrastiveLoss label 1 is", loss_func(a1,a2,1))
    print("ContrastiveLoss label 0 is", loss_func(a1,a2,0))
    print("totalLoss is")
    W = SiameseLoss(a,1)
    print("label 1 is")
    print(W)
    print("label 0 is")
    W = SiameseLoss(a,1)
    print(W)

def test():
    # Y = torch.FloatTensor([[1,2,3],[2,3,4],[3,4,5]]).numpy()
    # W = cosine_similarity(Y,Y)
    # print(W)
    # Y = torch.FloatTensor([[1,2,3],[2,3,4],[3,4,5]])
    Y = torch.rand(3,3)
    print(Y)
    # W = torch.sort(-Y,dim=1)
    # indices = torch.randperm(2)
    # print(indices)
    # print(Y[indices])
    # print(Y)
    # print(W)
    print(thrC(Y,0.5))


def testOrthonorm():
    a = torch.randn(5,4)
    print("print input")
    print(a)
    print("w")
    w = orthonorm(a)
    print(w)
    print("wt*w")
    print(torch.transpose(w,1,0).mm(w))

def testKmeans():
    input = torch.FloatTensor([[1,2,3],[2,3,4],[3,4,5]])
    k = 2
    cluster_center,labels = kmeans(k, input)
    print(labels)

def testTorchKtop():
    batch_size = 20
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)
    images,labels = next(iter(train_loader))
    data = images.view(images.shape[0],-1)
    D = data.numpy()
    W = cosine_similarity(D,D)
    print(W)
    W = torch.FloatTensor(W)
    nn = torch.topk(W,batch_size)
    qq = torch.randn(batch_size,batch_size)
    nnInd = nn[1]
    qq[nnInd] = 0
    # nn = getKneibor(X=data,n_nbrs=batch_size)
    print(nnInd)
    print(qq)
    show_img(images,batch_size)

def testSelfExpress():
    batch_size = 100
    train_dataset = downloadData(True)
    train_loader = loadData(train_dataset, batch_size, True)

    images, labels = next(iter(train_loader))
    images = get_variable(images)
    images = F.normalize(images)
    labels = get_variable(labels)
    
    module = TrainSelfExpression(images)
    C,Z,ZC,encode,decode = module(images)
    C = thrC(C,0.3)
    print("len C!=0")
    print(len(C[C!=0]))
    C[C > 0] = 1
    C = C.int()
    print("len C > 0")
    print(len(C[C>0]))
    R = getKneibor(lables=labels)
    R[R == 0] = -1
    Dif = C - R
    print("len R>0")
    print(len(R[R>0]))
    err1 = len(Dif[Dif==2])
    print("C is 1 R is 0")
    print(err1)
    err2 = len(Dif[Dif==-1])
    print("C is 0 R is 1")
    print(err2)
    err = err1 + err2

    succ = len(Dif[Dif==0])
    print("R is 1")
    print(len(R[R==1]))
    print("C == R")
    print(succ)

    rate = float(succ) / (err + succ)
    print(rate)
    show_img(decode,batch_size)

def Test():
    batch_size = 100
    labSize = 10
    layers = 3
    num_epochs = 30
    test_dataset = downloadData(True)
    test_loader = loadData(test_dataset, batch_size, True)
    cnn = CNN2()
    cnn.load_state_dict(torch.load("./module/CNN2.pkl"))
    # norm_k_os = SpectralNetNorm()
    # norm_k_os.load_state_dict(torch.load("./module/SpectralNetNorm_Tanh.pkl"))

    totAcc = 0
    for i,(images,labels) in enumerate(test_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        images = F.normalize(images)
        show_img(images,batch_size)
        encode,decode = cnn(images)
        show_img(decode,batch_size)
        
def Test():
    batch_size = 100
    labSize = 10
    layers = 3
    num_epochs = 30
    test_dataset = downloadData(True)
    test_loader = loadData(test_dataset, batch_size, True)

    totAcc = 0
    for i,(images,labels) in enumerate(test_loader):
        images = get_variable(images)
        labels = get_variable(labels)
        images = F.normalize(images)
        show_img(images,batch_size)
        encode,decode = cnn(images)
        show_img(decode,batch_size)

def TestEmbed():
    # batch_size = 1024
    # labSize = 10
    # layers = 3
    # num_epochs = 30
    # test_dataset = downloadData(True)
    # test_loader = loadData(test_dataset, batch_size, True)
    # images,labels = next(iter(test_loader))
    # x = embed_data(images.numpy())
    cnn = CNN()
    cnn.load_state_dict(torch.load('./pretrain_weight/ae_mnist_weights.h5'))
if __name__ == "__main__":
    # testAcc()
    # testCNN()
    # testGetAMatrix()
    # testSelfExpression()
    # testKnn()
    # testGetKneibor()
    testSiamese()
    # testsquared_distance()
    # testOrthonorm()
    # test()
    # testKmeans()
    # testTorchKtop()
    # testSelfExpress()
    # Test()
    # TestEmbed()