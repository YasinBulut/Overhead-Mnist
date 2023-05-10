import matplotlib.pyplot as plt
import numpy as np

#NEURAL NETWORK CLASS DEFİNİTİON
class neuralNetwork:
    #İNİTİALİSE THE NEURAL NETWORK 
    def __init__(self,inputNodes, hiddenNodes, outputNodes, learningRate):
        #SET NUMBER OF NODES İN EACH İNPUT, HİDDEN, OUTPUT LAYER     
        self.inodes=inputNodes  #GİRİSLER
        self.hnodes=hiddenNodes #ARA KATMAN 
        self.onodes=outputNodes #CIKISLAR 

        self.akj = np.random.uniform(size=self.inodes * self.hnodes, low=-1,high=1).reshape(self.hnodes,self.inodes)
        self.ajm = np.random.uniform(size=self.hnodes * self.onodes, low=-1,high=1).reshape(self.hnodes,self.onodes)


        self.bj=np.random.uniform(size=self.hnodes, low=-1,high=1).reshape(self.hnodes, 1)
        self.bm=np.random.uniform(size=self.onodes, low=-1,high=1).reshape(self.onodes, 1)
        
        #LEARNİNG RATE --> ÖĞRENME KATSAYISI TANIMLAMASI YAPILIR !
        self.lr=learningRate
    #ACTİVATİON FUNCTİON İS SİGMOİD    
    def activation_func(self,x):
        return 1 / (1+np.exp(-x))

    #NET HESABI 
    def query(self, input_list):
        #CONVERT İNPUT LİST TO 2D ARRAY --> 2 BOYUTLU GİRİS LİSTESİ MATRİSİ    
        inputs = np.array(input_list, ndmin=2).T  #(T) TRANSPOZ ALMA İÇİN KULLANILDI!

        #CALCULATE SİGNALS İNTO HİDDEN LAYER
        Net_j = np.dot(self.akj, inputs) + self.bj
        Ç_j = self.activation_func(Net_j)

        #CALCULATE SİGNALS İNTO OUTPUT LAYER
        Net_m = np.transpose(np.dot(np.transpose(Ç_j), self.ajm)) + self.bm
        Ç_m = self.activation_func(Net_m)

        return Ç_m
    #İLERİYE DOĞRU HESAPLAMA BURADA BASLAR ! !
    def train(self, input_list, target_list):
        #CONVERT İNPUT LİST TO 2D ARRAY
        inputs = np.array(input_list, ndmin=2).T
        B_ç = np.array(target_list, ndmin=2).T

        #CALCULATE SİGNALS İNTO HİDDEN LAYER
        Net_j = np.dot(self.akj, inputs) + self.bj
        Ç_j = self.activation_func(Net_j)

        #CALCULATE SİGNALS İNTO OUTPUT LAYER
        Net_m = np.transpose(np.dot(np.transpose(Ç_j), self.ajm)) + self.bm 
        Ç_m = self.activation_func(Net_m)
    #İLERİYE DOGRU HESAPLAMA BURADA BİTTİ ! !
    
    #GERİYE DOGRU YAYILIMDAN DEVAM ! ! --> BACKPROPAGATİON   
        E_m = B_ç - Ç_m

        delta_m = Ç_m * (1-Ç_m) * E_m
        hidden_error = np.dot(self.ajm, delta_m)

        
        delta_ajm = np.dot(Ç_j, self.lr * np.transpose(delta_m))  #burada momentum katsatısı yok fakat sen sonradan kendin eklemeyi unutma kendi projende
        self.ajm += delta_ajm 

        delta_bm = self.lr * delta_m
        self.bm += delta_bm

        delta_j = Ç_j * (1-Ç_j) * hidden_error

        delta_akj = np.dot(self.lr * delta_j, np.transpose(inputs))     ##Çk=Gk oldugundan çıktılar yerine inputs yazdık!!
        self.akj += delta_akj

        delta_bj = self.lr * delta_j
        self.bj += delta_bj

    #AĞIRLIKLARIN YAZDIRILMASI ! !
    def print_weight(self):
        print("akj:", self.akj)
        print("ajm:", self.ajm)
        print("bj:", self.bj)
        print("bm:", self.bm)      

# SİNİR AĞININ YAPILANDIRMA AYARLARI
input_nodes = 784
hidden_nodes = 98
output_nodes = 10

learning_rate = 0.784

n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

#LANDA --> AĞIRLIKLARIN DEĞİŞİM MİKTARLARINA ETKİ EDER / ÇALIŞMA SÜRESİNE ETKİ ETMEZ ! !
#EĞİTİM VERİLERİNİN ALINMASI (load the mnist training data CSV file into a list)
training_data_file = open("C:/Users/Yasin/Desktop/Proje/train.csv","r")
training_data_list = training_data_file.readlines()
training_data_file.close()

#AĞIN EĞİTİME BASLANMASI 
epochs = 100

for e in range(epochs):
    for record in training_data_list:
        all_values = record.split(',')  #veri setinde datalar yan yana virgülle ayrılmıstı onları virgüllerden ayır!!
        inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
        #[1:] neden birden basladı ilk data degeri bize sonucu verecek ondan dolayı 1 den baslatıyoruz data okuması için! ilk data alınmaz!
        #1.data (n=0) da ki data bizim etiket değerimizdir 
        B_ç = np.zeros(output_nodes) + 0.01

        B_ç[int(all_values[0])] = 0.99  #bç listesi 10 adet degerden olusacak 0-9 arası 7.data degeri 0.99 olacak ve bu bize islemin sonucunu verecektir 0.99 olan indeks!
        n.train(inputs,B_ç)

#test verisinin yüklenmesi
test_data_file = open("C:/Users/Yasin/Desktop/Proje/test.csv","r")
test_data_list = test_data_file.readlines()
test_data_file.close()        

count = 0

for record in test_data_list:
    all_values = record.split(",")

    inputs = (np.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01

    B_ç = np.zeros(output_nodes) + 0.01

    B_ç[int(all_values[0])] = 0.99
    result = n.query(inputs)

    print("Cikti(Ç_m): %d, Beklenen(B_ç): %d" % (np.argmax(result), int(all_values[0])))

    if np.argmax(result) == int(all_values[0]):
        count += 1

print("Basari: %f" % (count / len(test_data_list)))
                #KODUN ÇALIŞMASI İÇİN QUERY FONKSİYONUNU ÇAĞIR VE İNPUTS DEGERLERİ VER !

#VERİ SETİNİN GÖRSELLEŞTİRİLMESİ !! 
all_values = test_data_list[0].split(',')
image_array = np.asfarray(all_values[1:]).reshape((28,28))
plt.imshow(image_array, cmap='Greys', interpolation='None')

plt.show()