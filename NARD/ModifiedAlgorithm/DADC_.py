
import numpy as np

class DADC:
    MAX = 1000000

    maxdist = 0 

    def __init__(self,points,k_percent,cfd_threshold):
        self.points_=points
        self.result_=[]
        self.percent_=k_percent
        self.cfd_threshold_=cfd_threshold
    #1 main function of DADC
    def runAlgorithm(self): 
        #1) load input data
        points=self.points_
        length = len(points)

        #2) compute rho density and delta distance
        ll, dist = self.getDistance(points)    #compute distances
        self.maxdist = np.max(ll)   #Global maximum distance 
        #kls: set of neighbors; DD: domain density; DAD: domain-adaptive density; delta: delta distance       
        kls, DD, DAD,delta = self.DADCMethod(ll, dist, length)           
        #3) identify cluster centers
        # self.pf.printRhoDelta(DAD,delta)  # print clustering decision graph
        centers = self.identifyCenters(DAD, delta, length) 
        
        #4) assign the remaining points
        result = self.assignDataPoint(dist, DAD, centers, length)

        #5) clusterEnsemble
        if (np.max(result)>0):        
            cfd_threshold = self.cfd_threshold_
            cfd, result = self.clusterEnsemble(points, result, kls, DD, cfd_threshold)
            while(np.max(cfd)> cfd_threshold and np.max(cfd)!=1):
                cfd, result = self.clusterEnsemble(points, result, kls, DD, cfd_threshold)
                #print(result)                  
                # self.pf.printPoltLenged(points,result)

        self.result_ = result

    #2 compute rho density and delta distance
    def DADCMethod(self, ll, dist, length):
        #1) compute the cutoff distance
        percent = self.percent_    # percent of  Number of neighbors
        k =int(length * percent / 100)  # Number of neighbors
        print ("Number of neighbors (k): ",k)
        #2) compute KNN-distance and KNN-density
        #kls: set of neighbors; kDist: KNN-distance; kDen: KNN-density
        kls, kDist, kDen = self.getKNNDensity(dist, k, length)
        #3) compute domain density and domain-adaptive density
        DD = self.getDomainDensity(kls, kDen, kDist)  
        DAD = self.getDomainAdaptiveDensity(DD, dist,length)  #计算相对域密度
        #4) compute delta distance
        #delta = self.computDeltaDistance(DAD,dist,length)
        delta = self.computDeltaDistance2(DAD, kls, kDist, dist)
        return kls, DD, DAD, delta   
       
    #3 compute distances among data points
    def getDistance(self,points):
        length =len(points)
        dist = np.zeros((length, length))
        ll = []
        for i in range(length-1):
            for j in range(i+1, length):
                dd = np.linalg.norm(points[i] - points[j])
                dist[i][j] = dist[j][i] = dd
                ll.append(dd)
        ll = np.array(ll)
        # self.fo.writeData(dist,self.fileurl +'distance.csv')
        return ll,dist
    
    
    #4 compute KNN-distance and KNN-density    
    def getKNNDensity(self,dist, k, length): 
        kls =np.zeros((length, k),dtype = np.integer)  #set of neighbors
        kDist = np.zeros((length, k)) #KNN distance
        kDen =  np.zeros((length, 3)) #KNN density
        for i in range(length):
            ll = dist[i]   
            sortedll = np.sort(ll)      
            kDist[i] = sortedll[1:k+1]
            j = 0
            kls_temp= []
            while j < k:
                temp = np.where(ll==kDist[i][j])   
                temp2 = temp[0]  
                j = j+len(temp2)
                kls_temp.extend(list(temp2))         
            kls[i] =kls_temp[0:k]   #get the first k nearest neighbors for each point    
            kDen[i][0] = 1 / np.average(kDist[i]) if (np.average(kDist[i])!=0) else 0 #get KNN-density
            kDen[i][1] = np.average(kDist[i]) #get the average distance of the k neighbors (KNN-distance)
            kDen[i][2] = sortedll[k]      #get the distance from xi to the k-th neighbor
        #print("knn list: ",kls)
        #print("Kdist: ",kDist)
        #print("knn density: ", kDen)       
        #self.fo.writeData(kls, self.fileurl +'kls.csv')     
        #self.fo.writeData(kDist, self.fileurl +'KNNDistance.csv') 
        #self.fo.writeData(kDen, self.fileurl +'KNNDensity.csv')    
        return kls, kDist, kDen
    
    
    #5 compute Domain density 
    # inputs: kls: set of neighbors, kDen: KNN-distance and KNN-density (3d)
    # output: domein density
    def getDomainDensity(self, kls, kDen, kDist): 
        DD = []
        for i in range(len(kls)):
            #di = sum(kDist[i])/len(kDist[i])
            Di = kDen[i][0] #get KNN-density of xi 
            for j in kls[i]: # for each neighbor
                #Di =Di + kDen[j][0] # method 1: directly add each neighbor's KNN-density
                if(kDen[j][0] <= kDen[i][0] and kDist[i,np.where(kls[i]==j)]!=0): #method 2: add each neighbor's KNN-density with its weight
                    wkDenj = kDen[j][0] * (1/kDist[i,np.where(kls[i]==j)])  #wkDenj is an array
                    Di =Di + wkDenj[0] 
            DD.append(Di)
        # self.fo.writeData(DD, self.fileurl +'DomainDensity.csv')
        # self.pf.printPolt3(DD)  # print graph for rho
        return DD  
    
    
    #6 computer domain-adaptive density 
    #  Eliminate the difference in domain density between varying-density regions (VDD)
    def getDomainAdaptiveDensity(self, DD, dist, length):
        DAD = np.ones((length, 1))* self.MAX
        maxDensity = np.max(DD)
        for i in range(length):
            if DD[i] < maxDensity:
                for j in range(length):
                    if DD[j] > DD[i] and dist[i][j] < DAD[i]:
                        DAD[i] = DD[i] * (dist[i][j]/self.maxdist) #WDD[i] * dist[i][j]
            else:
                DAD[i] = 0.0
                for j in range(length):
                    if dist[i][j] > DAD[i]:
                        DAD[i] = DD[i] * dist[i][j]
        # self.fo.writeData(DAD, self.fileurl +'DomainAdaptiveDensity.csv')
        # self.pf.printPolt3(DAD)  # print graph for rho
        return DD  
  
       
    #7 compute Delta distance
    def computDeltaDistance(self, rho, dist, length): 
        delta = np.ones((length, 1)) * self.MAX
        maxDensity = np.max(rho)
        for i in range(length):
            if rho[i] < maxDensity:
                for j in range(length):
                    if rho[j] > rho[i] and dist[i][j] < delta[i]:
                        delta[i] = dist[i][j]
            else:
                delta[i] = 0.0
                for j in range(length):
                    if dist[i][j] > delta[i]:
                        delta[i] = dist[i][j]
        # self.fo.writeData(delta, self.fileurl +'DADC-Delta.csv')
        return delta


    #7 compute Delta distance
    def computDeltaDistance2(self, rho, kls, kDist, dist): 
        length = len(kls)  
        delta = np.ones((length, 1)) * self.MAX
        for i in range(length):
            rho_knn =[]
            for j in kls[i]:  #get the domain-adaptive density of neighbors
                rho_knn.append(rho[j])
            rho_knn2 = np.sort(rho_knn)  
            #the point is a domain density peak, if its domain-adaptive density is greater than that of all neighbors
            if rho[i] >= rho_knn2[-1]:
                delta[i] =np.max(dist[i])
            else:
                delta[i] = np.min(dist[i])     
        # self.fo.writeData(delta, self.fileurl +'DADC-Delta.csv')
        return delta
    

    #8 initial cluster censter self-identification
    def identifyCenters(self, rho, delta, length):
        #the value of critical point in clustering decision graph
        thRho = np.max(rho)/2
        thDel = np.max(delta)/4  
 
        centers = np.ones(length, dtype=np.int) * (-1)
        cNum = 0
        for i in range(length):
            if rho[i] > thRho and delta[i] > thDel:
                centers[i] = cNum
                cNum = cNum + 1
        # print("Number of initial cluster centers",cNum)
        return centers        
 
 
    #9 assign the remaining points to the corresponding cluster center
    def assignDataPoint(self, dist,rho, result, length):
        for i in range(length):
            dist[i][i] = self.MAX

        for i in range(length):
            if result[i] == -1:
                result[i] = self.nearestNeighbor(i,dist, rho, result, length)
            else:
                continue
        return result
      
      
    #10 Get the nearest neighbor with higher rho density for each point        
    def nearestNeighbor(self,index, dist, rho, result,length):
        MAX = 1000000
        dd = MAX
        neighbor = -1
        for i in range(length):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        if result[neighbor] == -1:
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result, length)
        return result[neighbor]


    #11 fragmented cluster self-ensemble 
    def clusterEnsemble(self, points, result, kls, DD, cfd_threshold):
        cNum = np.max(result)+1  #Number of clusters
        ids = self.getIDS(result, DD)  # compute inter-cluster density similarity 
        ccd = self.getCCD(result, kls) # compute cluster crossover degree  
        cds = self.getCDS(result, DD)  # compute cluster density stability
        cfd = self.getCFD(ids, ccd, cds) # compute the cluster fusion degree
        
        for i in range(cNum):
            for j in range(i+1, cNum):
                if cfd[i,j] > cfd_threshold:
                    #ensemble clusters ci and cj                    
                    ks = list(k for k in range(len(result)) if result[k]==j)
                    for k in ks:
                        result[k] =i
                    ks2 = list(k for k in range(len(result)) if result[k]>j)
                    for k in ks2:
                        result[k]= result[k]-1
                                
        return cfd, result            
        
    #12 compute inter-cluster density similarity    
    def getIDS(self, result, DD):   
        cNum = np.max(result)+1  #Number of clusters    
        ids = np.zeros((cNum, cNum))
        
        #1) get the cluster density
        Cdens = np.zeros(cNum)  
        for i in range(cNum):
            Cdens[i] = np.average(list(DD[j] for j in list(j for j in range(len(result)) if result[j]==i)))
            
        for i in range(cNum):
            for j in range(i+1, cNum):
                ids[i,j] = ids[j,i] = (np.sqrt(Cdens[i] * Cdens[j])*2)/(Cdens[i]+Cdens[j]) 
        return ids            
           
    #13 compute cluster crossover degree    
    def getCCD(self, result, kls):
        pointNum = len(result)   #Number of data points
        cNum = np.max(result)+1  #Number of clusters
        #1) crossover degree for each point
        c = np.zeros((pointNum, cNum))  
        for i in range(pointNum):
            ci = result[i] #the cluster of xi
            cn =list(result[j] for j in kls[i])   #the cluster of each neighbor
            for j in range(cNum):
                cj = cn.count(j)   #number of neighbors in j-th cluster
                ci2= cn.count(ci)  #number of neighbors in i-th cluster 
                c[i,j] = (np.sqrt(ci2 * cj)*2)/(ci2+cj) if (cj!=0 and ci2!=0) else 0
        #print("crossover degree of each point:",c)
        
        #2)compute cluster crossover degree (CCD)
        ccd = np.zeros((cNum, cNum))
        for i in range(cNum):
            for j in range(i+1, cNum):
                n = list(result).count(i)+list(result).count(j)  #Number of points in ci and cj
                ccd[i,j] = ccd[j,i] = ((sum(list(c[k,j] for k in range(len(result)) if result[k]==i))   #c(x, i->j) points in cluster i
                                        + sum(list(c[k,i] for k in range(len(result)) if result[k]==j)))/n)  #c(x, j->i) points in cluster j            
        return ccd   
            
    #14 compute cluster density stability
    def getCDS(self, result, DD):
        cNum = np.max(result)+1  #Number of clusters
        #1) compute the cds of each cluster
        da = np.zeros(cNum)
        for i in range(cNum):
            li = list(k for k in range(len(result)) if result[k]==i) #points in cluster ci
            den_avg = np.average(list(DD[k] for k in li))  #the average domain density of cluster ci
            da[i] = 1/np.sqrt(np.sum(list(np.square(DD[k] - den_avg) for k in li))) if (np.sum(list(np.square(DD[k] - den_avg) for k in li))!=0) else 0
                    
        #2) compute the cds among clusters
        cds = np.zeros((cNum, cNum))
        for i in range(cNum):
            for j in range(i+1, cNum):
                li = list(k for k in range(len(result)) if (result[k]==i or result[k]==j)) #points in cluster ci
                denavg_ab = np.average(list(DD[k] for k in li))  #the average domain density of cluster ci 
                den_ab =  np.sqrt(np.square(len(li))/np.sum(list(np.square(DD[k] - denavg_ab) for k in li)))
                cds[i,j] = cds[j,i] = (2* den_ab)/(da[i]+da[j]) if (da[i] !=0 and da[j]!=0) else 0
        return cds
        
    #15 compute the cluster fusion degree
    def getCFD(self, ids, ccd, cds):
        cfd = (np.sqrt(3)/4)*(ids * ccd + ccd * cds + cds * ids)
        
        ids_max =np.max(ids)
        ccd_max =np.max(ccd)
        cds_max =np.max(cds)
        cfd_max = (np.sqrt(3)/4)*(ids_max * ccd_max + ccd_max * cds_max + cds_max * ids_max)
        cfd2=cfd/cfd_max
        # print(cfd2)
        return cfd2


class DADC_Auto_Adaptive:
    MAX = 1000000

    maxdist = 0

    def __init__(self, points, NARD):
        self.points_ = points
        self.result_ = []
        self.nard_=NARD
    # 1 main function of DADC
    def runAlgorithm(self):
        # 1) load input data
        points = self.points_
        length = len(points)

        # 2) compute rho density and delta distance
        ll, dist = self.getDistance(points)  # compute distances
        self.maxdist = np.max(ll)  # Global maximum distance
        # kls: set of neighbors; DD: domain density; DAD: domain-adaptive density; delta: delta distance
        # kls, DD, DAD, delta = self.DADCMethod(ll, dist, length)

        # 3) identify cluster centers
        # self.pf.printRhoDelta(DAD, delta)  # print clustering decision graph
        # centers = self.identifyCenters(DAD, delta, length)
        centers = np.ones(length, dtype=np.int) * (-1)

        for count,assistCenter in enumerate(self.nard_.centers_):
            centers[assistCenter]=count


        # 4) assign the remaining points
        result = self.assignDataPoint(dist,self.nard_.NARD_, centers, length)

        # 5) clusterEnsemble
        # if (np.max(result) > 0):
        #     cfd_threshold = 0.6
        #     cfd, result = self.clusterEnsemble(points, result, kls, DD, cfd_threshold)
        #     while (np.max(cfd) > cfd_threshold and np.max(cfd) != 1):
        #         cfd, result = self.clusterEnsemble(points, result, kls, DD, cfd_threshold)
        #         # print(result)
                # self.pf.printPoltLenged(points, result)

        self.result_ = result

    # 2 compute rho density and delta distance
    def DADCMethod(self, ll, dist, length):
        # 1) compute the cutoff distance
        percent = 5  # percent of  Number of neighbors
        k = int(length * percent / 100)  # Number of neighbors

        # 2) compute KNN-distance and KNN-density
        # kls: set of neighbors; kDist: KNN-distance; kDen: KNN-density
        kls, kDist, kDen = self.getKNNDensity(dist, k, length)

        # 3) compute domain density and domain-adaptive density
        DD = self.getDomainDensity(kls, kDen, kDist)
        DAD = self.getDomainAdaptiveDensity(DD, dist, length)  # 计算相对域密度

        # 4) compute delta distance
        # delta = self.computDeltaDistance(DAD,dist,length)
        delta = self.computDeltaDistance2(DAD, kls, kDist, dist)
        return kls, DD, DAD, delta

        # 3 compute distances among data points

    def getDistance(self, points):
        length = len(points)
        dist = np.zeros((length, length))
        ll = []
        for i in range(length - 1):
            for j in range(i + 1, length):
                dd = np.linalg.norm(points[i] - points[j])
                dist[i][j] = dist[j][i] = dd
                ll.append(dd)
        ll = np.array(ll)
        # self.fo.writeData(dist,self.fileurl +'distance.csv')
        return ll, dist

    # 4 compute KNN-distance and KNN-density
    def getKNNDensity(self, dist, k, length):
        kls = np.zeros((length, k), dtype=np.integer)  # set of neighbors
        kDist = np.zeros((length, k))  # KNN distance
        kDen = np.zeros((length, 3))  # KNN density
        for i in range(length):
            ll = dist[i]
            sortedll = np.sort(ll)
            kDist[i] = sortedll[1:k + 1]
            j = 0
            kls_temp = []
            while j < k:
                temp = np.where(ll == kDist[i][j])
                temp2 = temp[0]
                j = j + len(temp2)
                kls_temp.extend(list(temp2))
            kls[i] = kls_temp[0:k]  # get the first k nearest neighbors for each point
            kDen[i][0] = 1 / np.average(kDist[i]) if (np.average(kDist[i]) != 0) else 0  # get KNN-density
            kDen[i][1] = np.average(kDist[i])  # get the average distance of the k neighbors (KNN-distance)
            kDen[i][2] = sortedll[k]  # get the distance from xi to the k-th neighbor
        # print("knn list: ",kls)
        # print("Kdist: ",kDist)
        # print("knn density: ", kDen)
        # self.fo.writeData(kls, self.fileurl +'kls.csv')
        # self.fo.writeData(kDist, self.fileurl +'KNNDistance.csv')
        # self.fo.writeData(kDen, self.fileurl +'KNNDensity.csv')
        return kls, kDist, kDen

    # 5 compute Domain density
    # inputs: kls: set of neighbors, kDen: KNN-distance and KNN-density (3d)
    # output: domein density
    def getDomainDensity(self, kls, kDen, kDist):
        DD = []
        for i in range(len(kls)):
            # di = sum(kDist[i])/len(kDist[i])
            Di = kDen[i][0]  # get KNN-density of xi
            for j in kls[i]:  # for each neighbor
                # Di =Di + kDen[j][0] # method 1: directly add each neighbor's KNN-density
                if (kDen[j][0] <= kDen[i][0]):  # method 2: add each neighbor's KNN-density with its weight
                    wkDenj = kDen[j][0] * (1 / kDist[i, np.where(kls[i] == j)])  # wkDenj is an array
                    Di = Di + wkDenj[0]
            DD.append(Di)
        # self.fo.writeData(DD, self.fileurl +'DomainDensity.csv')
        # self.pf.printPolt3(DD)  # print graph for rho
        return DD

        # 6 computer domain-adaptive density

    #  Eliminate the difference in domain density between varying-density regions (VDD)
    def getDomainAdaptiveDensity(self, DD, dist, length):
        DAD = np.ones((length, 1)) * self.MAX
        maxDensity = np.max(DD)
        for i in range(length):
            if DD[i] < maxDensity:
                for j in range(length):
                    if DD[j] > DD[i] and dist[i][j] < DAD[i]:
                        DAD[i] = DD[i] * (dist[i][j] / self.maxdist)  # WDD[i] * dist[i][j]
            else:
                DAD[i] = 0.0
                for j in range(length):
                    if dist[i][j] > DAD[i]:
                        DAD[i] = DD[i] * dist[i][j]
        # self.fo.writeData(DAD, self.fileurl +'DomainAdaptiveDensity.csv')
        # self.pf.printPolt3(DAD)  # print graph for rho
        return DD

        # 7 compute Delta distance

    def computDeltaDistance(self, rho, dist, length):
        delta = np.ones((length, 1)) * self.MAX
        maxDensity = np.max(rho)
        for i in range(length):
            if rho[i] < maxDensity:
                for j in range(length):
                    if rho[j] > rho[i] and dist[i][j] < delta[i]:
                        delta[i] = dist[i][j]
            else:
                delta[i] = 0.0
                for j in range(length):
                    if dist[i][j] > delta[i]:
                        delta[i] = dist[i][j]
        # self.fo.writeData(delta, self.fileurl +'DADC-Delta.csv')
        return delta

    # 7 compute Delta distance
    def computDeltaDistance2(self, rho, kls, kDist, dist):
        length = len(kls)
        delta = np.ones((length, 1)) * self.MAX
        for i in range(length):
            rho_knn = []
            for j in kls[i]:  # get the domain-adaptive density of neighbors
                rho_knn.append(rho[j])
            rho_knn2 = np.sort(rho_knn)
            # the point is a domain density peak, if its domain-adaptive density is greater than that of all neighbors
            if rho[i] >= rho_knn2[-1]:
                delta[i] = np.max(dist[i])
            else:
                delta[i] = np.min(dist[i])
                # self.fo.writeData(delta, self.fileurl +'DADC-Delta.csv')
        return delta

    # 8 initial cluster censter self-identification
    def identifyCenters(self, rho, delta, length):
        # the value of critical point in clustering decision graph
        thRho = np.max(rho) / 2
        thDel = np.max(delta) / 4

        centers = np.ones(length, dtype=np.int) * (-1)
        cNum = 0
        for i in range(length):
            if rho[i] > thRho and delta[i] > thDel:
                centers[i] = cNum
                cNum = cNum + 1
        # print("Number of initial cluster centers",cNum)
        return centers

        # 9 assign the remaining points to the corresponding cluster center

    def assignDataPoint(self, dist, rho, result, length):
        for i in range(length):
            dist[i][i] = self.MAX

        for i in range(length):
            if result[i] == -1:
                result[i] = self.nearestNeighbor(i, dist, rho, result, length)
            else:
                continue
        return result

    # 10 Get the nearest neighbor with higher rho density for each point
    def nearestNeighbor(self, index, dist, rho, result, length):
        MAX = 1000000
        dd = MAX
        neighbor = -1
        for i in range(length):
            if dist[index, i] < dd and rho[index] < rho[i]:
                dd = dist[index, i]
                neighbor = i
        if result[neighbor] == -1:
            result[neighbor] = self.nearestNeighbor(neighbor, dist, rho, result, length)
        return result[neighbor]

    # 11 fragmented cluster self-ensemble
    def clusterEnsemble(self, points, result, kls, DD, cfd_threshold):
        cNum = np.max(result) + 1  # Number of clusters
        ids = self.getIDS(result, DD)  # compute inter-cluster density similarity
        ccd = self.getCCD(result, kls)  # compute cluster crossover degree
        cds = self.getCDS(result, DD)  # compute cluster density stability
        cfd = self.getCFD(ids, ccd, cds)  # compute the cluster fusion degree

        for i in range(cNum):
            for j in range(i + 1, cNum):
                if cfd[i, j] > cfd_threshold:
                    # ensemble clusters ci and cj
                    ks = list(k for k in range(len(result)) if result[k] == j)
                    for k in ks:
                        result[k] = i
                    ks2 = list(k for k in range(len(result)) if result[k] > j)
                    for k in ks2:
                        result[k] = result[k] - 1

        return cfd, result

        # 12 compute inter-cluster density similarity

    def getIDS(self, result, DD):
        cNum = np.max(result) + 1  # Number of clusters
        ids = np.zeros((cNum, cNum))

        # 1) get the cluster density
        Cdens = np.zeros(cNum)
        for i in range(cNum):
            Cdens[i] = np.average(list(DD[j] for j in list(j for j in range(len(result)) if result[j] == i)))

        for i in range(cNum):
            for j in range(i + 1, cNum):
                ids[i, j] = ids[j, i] = (np.sqrt(Cdens[i] * Cdens[j]) * 2) / (Cdens[i] + Cdens[j])
        return ids

        # 13 compute cluster crossover degree

    def getCCD(self, result, kls):
        pointNum = len(result)  # Number of data points
        cNum = np.max(result) + 1  # Number of clusters
        # 1) crossover degree for each point
        c = np.zeros((pointNum, cNum))
        for i in range(pointNum):
            ci = result[i]  # the cluster of xi
            cn = list(result[j] for j in kls[i])  # the cluster of each neighbor
            for j in range(cNum):
                cj = cn.count(j)  # number of neighbors in j-th cluster
                ci2 = cn.count(ci)  # number of neighbors in i-th cluster
                c[i, j] = (np.sqrt(ci2 * cj) * 2) / (ci2 + cj) if (cj != 0 and ci2 != 0) else 0
        # print("crossover degree of each point:",c)

        # 2)compute cluster crossover degree (CCD)
        ccd = np.zeros((cNum, cNum))
        for i in range(cNum):
            for j in range(i + 1, cNum):
                n = list(result).count(i) + list(result).count(j)  # Number of points in ci and cj
                ccd[i, j] = ccd[j, i] = ((sum(list(
                    c[k, j] for k in range(len(result)) if result[k] == i))  # c(x, i->j) points in cluster i
                                          + sum(list(c[k, i] for k in range(len(result)) if result[
                            k] == j))) / n)  # c(x, j->i) points in cluster j
        return ccd

        # 14 compute cluster density stability

    def getCDS(self, result, DD):
        cNum = np.max(result) + 1  # Number of clusters
        # 1) compute the cds of each cluster
        da = np.zeros(cNum)
        for i in range(cNum):
            li = list(k for k in range(len(result)) if result[k] == i)  # points in cluster ci
            den_avg = np.average(list(DD[k] for k in li))  # the average domain density of cluster ci
            da[i] = 1 / np.sqrt(np.sum(list(np.square(DD[k] - den_avg) for k in li))) if (
                        np.sum(list(np.square(DD[k] - den_avg) for k in li)) != 0) else 0

        # 2) compute the cds among clusters
        cds = np.zeros((cNum, cNum))
        for i in range(cNum):
            for j in range(i + 1, cNum):
                li = list(k for k in range(len(result)) if (result[k] == i or result[k] == j))  # points in cluster ci
                denavg_ab = np.average(list(DD[k] for k in li))  # the average domain density of cluster ci
                den_ab = np.sqrt(np.square(len(li)) / np.sum(list(np.square(DD[k] - denavg_ab) for k in li)))
                cds[i, j] = cds[j, i] = (2 * den_ab) / (da[i] + da[j]) if (da[i] != 0 and da[j] != 0) else 0
        return cds

    # 15 compute the cluster fusion degree
    def getCFD(self, ids, ccd, cds):
        cfd = (np.sqrt(3) / 4) * (ids * ccd + ccd * cds + cds * ids)

        ids_max = np.max(ids)
        ccd_max = np.max(ccd)
        cds_max = np.max(cds)
        cfd_max = (np.sqrt(3) / 4) * (ids_max * ccd_max + ccd_max * cds_max + cds_max * ids_max)
        cfd2 = cfd / cfd_max
        print(cfd2)
        return cfd2


def main():
    dadc = DADC()
    dadc.runAlgorithm()   #run the main function of CFSFDP 
    
       
if __name__ == "__main__":
    main()   