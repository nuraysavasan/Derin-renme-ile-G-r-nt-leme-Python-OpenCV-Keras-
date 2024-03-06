import glob
import os
import tensorflow as tf
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")#uyarıları kapat

imgs=glob.glob("./img/*.png") #"./img/" dizinindeki tüm ".png" dosyalarını eşleştirir ve imgs değişkenine atar.
widh= 125
height= 50

X=[]
Y=[]

for img in imgs:
    filename=os.path.basename(img)
    label=filename.split("_")[0] #dosya adının sıfırıncı indeksini alıyor
    im = np.array(Image.open(img).convert("L").resize((widh, height))) #yeniden boyutlandırıyoruz Görüntü açılır, gri tonlamalı yapılır ve belirtilen genişlik ve yüksekliğe yeniden boyutlandırılır
    im=im/255 #Görüntü piksel değerleri 0 ile 1 arasında ölçeklenir.
    X.append(im) #Ölçeklenmiş görüntü X listesine, etiket Y listesine eklenir.
    Y.append(label)
X=np.array(X)
X=X.reshape(X.shape[0],widh,height,1) #kaç resim olduğu,genişlik,yükseklik
#sns.countplot(Y)


#Etiketleri one-hot encode etmek için bir fonksiyon tanımlanır ve bu fonksiyon çağrılarak Y etiketleri dönüştürülür.
def onehot_labels(values):
    label_encoder=LabelEncoder() #girilen etiketlerin sayısal olarak kodlanmasını sağlar
    integer_encoded=label_encoder.fit_transform(values)  #fit_transform() metodu, girdi etiketlerini sayısal değerlere dönüştürür.
    onehot_encoder=OneHotEncoder(sparse_output=False) #one-hot encoding işlemi sonucunda yoğun (dense) düzenli  bir matris elde etmek için kullanılır.
    integer_encoded=integer_encoded.reshape(len(integer_encoded),1) #reshape() yöntemi, integer_encoded dizisini bir sütuna sahip bir matris haline getirir.
    onehot_encoded=onehot_encoder.fit_transform(integer_encoded) # integer_encoded'ı one-hot encoded forma dönüştürmek için kullanılır
    return onehot_encoded
Y=onehot_labels(Y) #Y adlı etiket listesini (label list) one-hot encoding'e dönüştürmek için onehot_labels fonksiyonunu çağırır ve dönüştürülmüş sonucu Y değişkenine atar.

#Veri kümesi, eğitim ve test alt kümelerine ayrılır.
train_X, test_X, train_Y, test_Y= train_test_split(X,Y, test_size=0.25,random_state=2)

#cnn model Sıralı bir model oluşturulur ve sinir ağı katmanları eklenir.
model=Sequential() #model oluşturulur
model.add(Conv2D(32,kernel_size=(3,3),activation="relu",input_shape=(widh,height,1)))
model.add(Conv2D(64,kernel_size=(3,3),activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2))) #Bu katman, görüntü boyutunu azaltmak için en büyük değeri seçerek havuzlama yapar.
model.add(Dropout(0.25)) #genelleştirme her öğrenme adımında rastgele birimlerin (nöronların) yüzde 25'inin pasifleştirilmesini sağlar.
model.add(Flatten()) #düzleştirme  tek boyutlu bir vektör elde eder.
model.add(Dense(128,activation="relu")) #Tam bağlı (fully connected) gizli katman. 128 nöron içerir ve ReLU aktivasyon fonksiyonu kullanır.
model.add(Dropout(0.4)) #Tekrar dropout uygulanır, bu sefer yüzde 40'ı pasifleştirilir.
model.add(Dense(3,activation="softmax")) #Tekrar dropout uygulanır, bu sefer yüzde 40'ı pasifleştirilir.


#Model derleme işlemi.
model.compile(loss="categorical_crossentropy",optimizer="Adam",metrics=["accuracy"])
model.fit(train_X,train_Y,epochs=35,batch_size=64) #Model eğitimi.

score_train=model.evaluate(train_X,train_Y)
print("eğitim doğruluğu",score_train[1]*100)

score_test=model.evaluate(test_X,test_Y)
print("test doğruluğu",score_test[1]*100)


#Modelin mimarisini JSON formatında bir değişkene kaydetme.
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model.to_json())

#open("model.json".write(model.to_json()))
model.save_weights("trex_weight_new.h5")





